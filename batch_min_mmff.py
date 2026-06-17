#!/usr/bin/env python
"""Batch protein-ligand energy minimization in tmol (MMFF94 charges, GPU-batched).

Given a newline-delimited .txt of complex PDB paths and ONE shared ligand SMILES,
for each complex: split protein/ligand (ProDy), rebuild ligand bond orders from the
SMILES template (RDKit), and Cartesian-minimize in tmol with MMFF94 ligand charges.

Design (see plan): the ligand is parameterized ONCE from the SMILES (MMFF94 charges
are topological, identical across complexes) into a single shared ParameterDatabase
+ PackedBlockTypes. Each complex then only supplies the ligand's pose coordinates and
is built with prepare_ligands=False, so all poses share one chem_db and can be stacked
with PoseStackBuilder.from_poses for batched GPU minimization.

Atom-ordering safeguard: SDF/MOL carries no atom names, so names are generated
deterministically (C1,N1,...) in RDKit/MOL-block order and never read from parsed
files. Each complex's ligand atoms are mapped to the canonical order via an RDKit
substructure match before coordinates are injected into the canonical ligand array.

Usage:
    python batch_min_mmff.py LIST.txt --smiles "<SMILES>" --out-dir DIR \
        [--res-name LIG] [--batch-size 8] [--apo] [--device cuda] \
        [--solvent-resnames HOH,WAT,NA,CL,...] [--resume] [--manifest m.csv]

Constraints: freeze atom categories during minimization via Cartesian coord_mask
(True=movable, False=frozen). Toggles combine freely:
    --freeze-backbone --freeze-sidechains --freeze-ligand-heavy --freeze-ligand-h
e.g. `--freeze-backbone --freeze-ligand-heavy` keeps the docked ligand pose and the
fold fixed while relaxing protein sidechains and the ligand's hydrogens.
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import torch

import prody as pr
from rdkit import Chem
from rdkit.Chem import AllChem
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.mol import MOLFile

import tmol
from tmol.database import ParameterDatabase
from tmol.ligand import prepare_ligands
from tmol.io.pose_stack_from_biotite import (
    canonical_form_from_biotite,
    _derived_types_for_param_db,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io.write_pose_stack_pdb import atom_records_from_pose_stack
from tmol.io.pdb_parsing import to_pdb

pr.confProDy(verbosity="none")

# tmol's .tmol (YAML) writer uses a custom _CompactDumper that cannot represent
# numpy scalar types. Atom names/charges that pass through biotite arrays arrive as
# numpy scalars, so register representers that coerce them to native Python types.
from tmol.ligand.params_io import _CompactDumper as _CompactDumper  # noqa: E402

for _nt, _rep in [
    (np.str_, lambda d, x: d.represent_str(str(x))),
    (np.float64, lambda d, x: d.represent_float(float(x))),
    (np.float32, lambda d, x: d.represent_float(float(x))),
    (np.int64, lambda d, x: d.represent_int(int(x))),
    (np.int32, lambda d, x: d.represent_int(int(x))),
    (np.bool_, lambda d, x: d.represent_bool(bool(x))),
]:
    _CompactDumper.add_representer(_nt, _rep)

DEFAULT_SOLVENT = [
    "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4",
    "ACT", "EDO", "GOL", "NO3", "BR", "IOD", "MN", "FE", "CU", "CD", "NI", "CO", "HG",
]


# --------------------------------------------------------------------------- #
# Stage 0: one-time ligand parameterization
# --------------------------------------------------------------------------- #
def _ligand_array_from_mol(mol, res_name):
    """RDKit Mol -> biotite AtomArray (+BondList) with generated unique names.

    Goes through a MOL block in memory (Chem.MolToMolBlock -> StringIO -> biotite).
    Names are assigned in RDKit/MOL-block atom order (C1, N1, ...).
    """
    molblock = Chem.MolToMolBlock(mol, kekulize=True)
    arr = MOLFile.read(io.StringIO(molblock)).get_structure()
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    n = arr.array_length()
    arr.res_name = np.array([res_name] * n)
    arr.res_id = np.array([1] * n)
    arr.chain_id = np.array(["L"] * n)
    arr.hetero = np.array([True] * n)
    counts: dict[str, int] = {}
    names = []
    for el in arr.element:
        el = str(el)
        counts[el] = counts.get(el, 0) + 1
        names.append(f"{el}{counts[el]}")
    arr.atom_name = np.array(names)
    return arr


def _canonical_mol(smiles, seed=0xF00D):
    """SMILES -> 3D Mol with explicit Hs (the canonical atom order + protomer)."""
    smimol = Chem.MolFromSmiles(smiles)
    if smimol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    canon = Chem.AddHs(smimol)
    if AllChem.EmbedMolecule(canon, randomSeed=seed) != 0:
        AllChem.EmbedMolecule(canon, randomSeed=seed, useRandomCoords=True)
    return smimol, canon


def _canonical_ligand_array(canon, res_name):
    """Canonical ligand AtomArray (+bonds) with generated names + MMFF94 charges."""
    props = AllChem.MMFFGetMoleculeProperties(canon)
    if props is None:
        raise ValueError("MMFF94 parameterization unavailable for this SMILES")
    charges = np.array([props.GetMMFFPartialCharge(i) for i in range(canon.GetNumAtoms())])
    lig_arr = _ligand_array_from_mol(canon, res_name)
    if lig_arr.array_length() != canon.GetNumAtoms():
        raise ValueError("biotite/RDKit ligand atom count mismatch")
    lig_arr.set_annotation("partial_charge", charges)
    return lig_arr


def build_ligand_params(smiles, res_name, out_tmol):
    """EXPENSIVE, run ONCE in the parent: parameterize the ligand and write a
    .tmol params file that workers cheaply load (skips per-worker RDKit typing)."""
    _, canon = _canonical_mol(smiles)
    lig_arr = _canonical_ligand_array(canon, res_name)
    prepare_ligands(
        lig_arr.copy(), param_db=ParameterDatabase.get_default(),
        charge_mode="auto", params_output=str(out_tmol),
    )


def load_ligand_ctx(smiles, res_name, device, tmol_path):
    """CHEAP per-worker: load the prebuilt .tmol and rebuild the matching context."""
    from tmol.ligand.params_file import inject_params_file
    smimol, canon = _canonical_mol(smiles)
    lig_arr = _canonical_ligand_array(canon, res_name)
    lig_db = inject_params_file(ParameterDatabase.get_default(), str(tmol_path))
    co, rts, pbt = _derived_types_for_param_db(lig_db, device)
    canon_el = [a.GetSymbol() for a in canon.GetAtoms()]
    return {
        "smimol": smimol, "canon": canon, "canon_el": canon_el, "lig_arr": lig_arr,
        "lig_db": lig_db, "co": co, "pbt": pbt, "res_name": res_name,
        "n_atoms": canon.GetNumAtoms(),
        "n_H": sum(1 for e in canon_el if e == "H"),
        "formal_charge": Chem.GetFormalCharge(canon), "device": device,
    }


# --------------------------------------------------------------------------- #
# Stage 1: per-complex extraction (fallible -> passthrough on error)
# --------------------------------------------------------------------------- #
def _protein_to_biotite(protein_sel):
    s = io.StringIO()
    pr.writePDBStream(s, protein_sel)
    arr = PDBFile.read(io.StringIO(s.getvalue())).get_structure(model=1)
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    return arr


def extract_complex(pdb_path, ctx, solvent_resnames):
    """Split + rebuild ligand chemistry; return combined + protein-only arrays."""
    ag = pr.parsePDB(str(pdb_path))
    if ag is None:
        raise ValueError("ProDy could not parse PDB")

    protein_sel = ag.select("not hetero and not element H")
    if protein_sel is None:
        raise ValueError("no protein atoms")

    sel = "hetero and not element H"
    if solvent_resnames:
        sel += " and not resname " + " ".join(solvent_resnames)
    het = ag.select(sel)
    if het is None:
        raise ValueError("no ligand hetero atoms (after solvent filter)")

    lig_stream = io.StringIO()
    pr.writePDBStream(lig_stream, het)
    mol = Chem.MolFromPDBBlock(lig_stream.getvalue(), proximityBonding=True)
    if mol is None:
        raise ValueError("RDKit could not parse ligand PDB block")
    mol = AllChem.AssignBondOrdersFromTemplate(ctx["smimol"], mol)
    mol = AllChem.AddHs(mol, addCoords=True)

    match = mol.GetSubstructMatch(ctx["canon"])
    if len(match) != ctx["n_atoms"]:
        raise ValueError("ligand does not fully match the SMILES template")
    re_el = [mol.GetAtomWithIdx(match[i]).GetSymbol() for i in range(len(match))]
    if re_el != ctx["canon_el"]:
        raise ValueError("element order mismatch after substructure match")
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(match[i])) for i in range(ctx["n_atoms"])],
        dtype=np.float32,
    )
    if not np.isfinite(coords).all():
        raise ValueError("non-finite ligand coordinates")

    lig = ctx["lig_arr"].copy()
    lig.coord = coords
    protein = _protein_to_biotite(protein_sel)
    comb = protein + lig
    return {"comb": comb, "protein": protein}


# --------------------------------------------------------------------------- #
# Stage 2: pose build (shared co/pbt -> all poses share chem_db)
# --------------------------------------------------------------------------- #
def build_pose(biotite_arr, ctx):
    cf = canonical_form_from_biotite(biotite_arr, ctx["device"], co=ctx["co"])
    return pose_stack_from_canonical_form(ctx["co"], ctx["pbt"], *cf)


# --------------------------------------------------------------------------- #
# Stage 3: batched minimize + write
# --------------------------------------------------------------------------- #
def _atomic_write(path, text):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _energies(sfxn, ps):
    return sfxn.render_whole_pose_scoring_module(ps)(ps.coords).detach().cpu().numpy()


def build_constraint_coord_mask(pose_stack, *, freeze_backbone, freeze_sidechains,
                                freeze_ligand_heavy, freeze_ligand_h):
    """Build a coord_mask (True=movable, False=frozen) for run_cart_min.

    Classifies every real atom into one of four categories and freezes the
    requested ones:
      - protein backbone   : polymeric block; mainchain_atoms (N/CA/C) PLUS the
                             carbonyl/terminal oxygen (O/OXT) and the backbone
                             hydrogens (H atoms bonded to a mainchain atom).
                             tmol's mainchain_atoms is only (N,CA,C), so we widen
                             it to the full backbone for a true "frozen backbone".
      - protein sidechain  : polymeric block, every real atom that is not backbone
      - ligand heavy       : non-polymeric block (polymer is None), non-H
      - ligand hydrogen    : non-polymeric block, H

    Returns a bool tensor [n_poses, max_n_atoms], or None if no toggle is set
    (so run_cart_min keeps its default all-atom behavior). Mirrors the
    block->atom scatter pattern of tmol.score.score_utils.build_sidechain_coord_mask
    and reuses PackedBlockTypes.atom_is_hydrogen.
    """
    if not (freeze_backbone or freeze_sidechains or freeze_ligand_heavy or freeze_ligand_h):
        return None

    n_poses, max_n_atoms, _ = pose_stack.coords.shape
    n_blocks = pose_stack.max_n_blocks
    max_n_block_atoms = pose_stack.max_n_block_atoms
    pbt = pose_stack.packed_block_types
    dev = pose_stack.device

    _, real_expanded = pose_stack.expand_coords()  # [n_poses, n_blocks, max_n_block_atoms]
    block_type_ind64 = pose_stack.block_type_ind64  # [n_poses, n_blocks]

    # per-(pose,block,atom) category flags
    is_backbone = torch.zeros((n_poses, n_blocks, max_n_block_atoms), dtype=torch.bool, device=dev)
    is_sidechain = torch.zeros_like(is_backbone)
    is_lig_heavy = torch.zeros_like(is_backbone)
    is_lig_h = torch.zeros_like(is_backbone)

    for bt_idx in range(pbt.n_types):
        bt_positions = block_type_ind64 == bt_idx  # [n_poses, n_blocks]
        if not bt_positions.any():
            continue
        rt = pbt.active_block_types[bt_idx]
        h_mask = pbt.atom_is_hydrogen[bt_idx, :max_n_block_atoms].bool()  # [max_n_block_atoms]
        mc_atoms = rt.properties.polymer.mainchain_atoms if rt.properties.polymer else None

        if mc_atoms is not None:  # polymeric (protein) residue
            mc_idx = {rt.atom_to_idx[a] for a in mc_atoms}
            h_np = h_mask.cpu().numpy()
            bb_idx = set(mc_idx)
            # carbonyl / terminal backbone oxygens (standard PDB names)
            bb_idx |= {i for i, at in enumerate(rt.atoms) if at.name in ("O", "OXT")}
            # backbone hydrogens: H atoms bonded to a mainchain atom (amide H, HA)
            for i, j in rt.bond_indices:
                if int(i) < len(h_np) and h_np[int(i)] and int(j) in mc_idx:
                    bb_idx.add(int(i))
            bb_mask = torch.zeros(max_n_block_atoms, dtype=torch.bool, device=dev)
            if bb_idx:
                bb_mask[torch.tensor(sorted(bb_idx), device=dev)] = True
            is_backbone[bt_positions] = bb_mask
            is_sidechain[bt_positions] = ~bb_mask
        else:  # non-polymeric (ligand) residue
            is_lig_heavy[bt_positions] = ~h_mask
            is_lig_h[bt_positions] = h_mask

    # movable starts as all real atoms; subtract frozen categories
    movable = real_expanded.clone()
    if freeze_backbone:
        movable &= ~is_backbone
    if freeze_sidechains:
        movable &= ~is_sidechain
    if freeze_ligand_heavy:
        movable &= ~is_lig_heavy
    if freeze_ligand_h:
        movable &= ~is_lig_h

    # Scatter block-atom layout [n_poses, n_blocks, max_n_block_atoms] into coord
    # space [n_poses, max_n_atoms]. Padding slots (local index >= a block's real
    # atom count) compute flat indices that fall inside the NEXT block's real
    # coordinates, so including them would make scatter_ collide and (on CUDA)
    # race non-deterministically. Real atoms pack contiguously and uniquely, so
    # route every non-real / out-of-range slot to an extra dustbin column that we
    # discard, leaving only collision-free real writes.
    atom_local_idx = (
        torch.arange(max_n_block_atoms, device=dev).view(1, 1, -1).expand(n_poses, n_blocks, -1)
    )
    flat_idx = (pose_stack.block_coord_offset64.unsqueeze(2) + atom_local_idx).reshape(n_poses, -1)
    movable_flat = movable.reshape(n_poses, -1)
    real_flat = real_expanded.reshape(n_poses, -1)

    keep = real_flat & (flat_idx < max_n_atoms)
    flat_idx = torch.where(keep, flat_idx, torch.full_like(flat_idx, max_n_atoms))

    coord_mask_ext = torch.zeros((n_poses, max_n_atoms + 1), dtype=torch.bool, device=dev)
    coord_mask_ext.scatter_(1, flat_idx, movable_flat & keep)
    return coord_mask_ext[:, :max_n_atoms]


def _write_poses_per_model(pose_stack, stems, suffix, out_dir):
    records = atom_records_from_pose_stack(pose_stack)
    for i, stem in enumerate(stems):
        # the structured array's "model" field holds strings ("1", "2", ...)
        sub = records[records["model"] == str(i + 1)].copy()
        sub["model"] = "1"
        _atomic_write(out_dir / f"{stem}{suffix}.pdb", to_pdb(sub))


def _cart_min(stacked, sfxn, constraints):
    """run_cart_min with an optional freeze mask. Returns (minimized_pose, status).

    status is "" normally, or "noop_all_frozen" when the constraints leave zero
    movable atoms (we skip the optimizer and return the input pose unchanged).
    """
    coord_mask = build_constraint_coord_mask(stacked, **constraints) if constraints else None
    if coord_mask is not None and not bool(coord_mask.any()):
        return stacked, "noop_all_frozen"
    return tmol.run_cart_min(stacked, sfxn, coord_mask), ""


def minimize_and_write(batch, ctx, sfxn, suffix, out_dir, manifest, passthrough_on_fail,
                       constraints=None):
    """batch: list of dicts {stem, pose, raw}. Batched min, fall back per-pose."""
    if not batch:
        return
    device = ctx["device"]
    stems = [b["stem"] for b in batch]
    poses = [b["pose"] for b in batch]
    try:
        stacked = PoseStackBuilder.from_poses(poses, device)
        e0 = _energies(sfxn, stacked)
        mn, status = _cart_min(stacked, sfxn, constraints)
        e1 = _energies(sfxn, mn)
        _write_poses_per_model(mn, stems, suffix, out_dir)
        tag = f"minimized{suffix}" if not status else f"{status}{suffix}"
        for i, b in enumerate(batch):
            manifest.append((b["stem"], tag, ctx["n_H"],
                             ctx["formal_charge"], float(e0[i]), float(e1[i])))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] batch {suffix} of {len(batch)} failed "
              f"({type(exc).__name__}: {exc}); isolating per-pose.", file=sys.stderr)
        for b in batch:
            try:
                single = PoseStackBuilder.from_poses([b["pose"]], device)
                e0 = float(_energies(sfxn, single)[0])
                mn, status = _cart_min(single, sfxn, constraints)
                e1 = float(_energies(sfxn, mn)[0])
                _write_poses_per_model(mn, [b["stem"]], suffix, out_dir)
                tag = f"minimized{suffix}" if not status else f"{status}{suffix}"
                manifest.append((b["stem"], tag, ctx["n_H"],
                                 ctx["formal_charge"], e0, e1))
            except Exception as exc2:  # noqa: BLE001
                if passthrough_on_fail and b.get("raw") is not None:
                    _atomic_write(out_dir / f"{b['stem']}{suffix}.pdb", b["raw"])
                    manifest.append((b["stem"], f"passthrough{suffix}", "", "", "", ""))
                else:
                    manifest.append((b["stem"], f"failed{suffix}:{type(exc2).__name__}",
                                     "", "", "", ""))
                print(f"[warn] {b['stem']}{suffix} failed: {exc2}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def read_list(list_path):
    paths = []
    for line in Path(list_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(line)
    return paths


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("list", help="newline-delimited .txt of complex PDB paths")
    ap.add_argument("--smiles", required=True, help="shared ligand SMILES (protomer source of truth)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res-name", default="LIG")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--apo", action="store_true", help="also minimize protein alone -> _apo.pdb")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", "-j", type=int, default=1,
                    help="parallel worker processes; each pinned to one GPU (multi-GPU sharding). "
                         "Default 1. Set to the number of GPUs you want to use.")
    ap.add_argument("--solvent-resnames", default=",".join(DEFAULT_SOLVENT),
                    help="comma-separated resnames to drop from the hetero selection")
    ap.add_argument("--resume", action="store_true", help="skip complexes whose outputs exist")
    ap.add_argument("--manifest", default=None, help="optional CSV path for per-complex status")
    cg = ap.add_argument_group(
        "constraints (freeze atom categories during minimization; combine freely)")
    cg.add_argument("--freeze-backbone", action="store_true",
                    help="freeze protein backbone (mainchain) atoms")
    cg.add_argument("--freeze-sidechains", action="store_true",
                    help="freeze protein sidechain atoms")
    cg.add_argument("--freeze-ligand-heavy", action="store_true",
                    help="freeze ligand heavy atoms (let only its hydrogens move)")
    cg.add_argument("--freeze-ligand-h", action="store_true",
                    help="freeze ligand hydrogen atoms")
    return ap.parse_args(argv)


def constraints_from_args(args):
    """The four freeze toggles as a dict for build_constraint_coord_mask."""
    return dict(
        freeze_backbone=args.freeze_backbone,
        freeze_sidechains=args.freeze_sidechains,
        freeze_ligand_heavy=args.freeze_ligand_heavy,
        freeze_ligand_h=args.freeze_ligand_h,
    )


def run_shard(paths, args, device_str, tmol_path, progress_queue=None):
    """Process one shard of complexes on one device. Returns manifest rows.

    Self-contained so it can run in a spawned worker process (multi-GPU sharding):
    each worker cheaply loads the prebuilt ligand .tmol and owns its CUDA device.
    ``progress_queue`` (optional): a mp Queue; one item is put per input path
    finished so the parent can drive a single tqdm bar across all shards.
    """
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solvent = [s for s in args.solvent_resnames.split(",") if s]

    ctx = load_ligand_ctx(args.smiles, args.res_name, device, tmol_path)
    sfxn = tmol.beta2016_score_function(device, param_db=ctx["lig_db"])
    constraints = constraints_from_args(args)

    holo_batch, apo_batch, manifest = [], [], []

    def flush(force=False):
        if force or len(holo_batch) >= args.batch_size:
            minimize_and_write(holo_batch, ctx, sfxn, "_holo", out_dir, manifest, True,
                               constraints)
            holo_batch.clear()
        if args.apo and (force or len(apo_batch) >= args.batch_size):
            minimize_and_write(apo_batch, ctx, sfxn, "_apo", out_dir, manifest, False,
                               constraints)
            apo_batch.clear()

    for path in paths:
        stem = Path(path).stem
        try:
            if args.resume and (out_dir / f"{stem}_holo.pdb").exists() and \
                    (not args.apo or (out_dir / f"{stem}_apo.pdb").exists()):
                continue
            raw = None
            try:
                raw = Path(path).read_text()
                ex = extract_complex(path, ctx, solvent)
                holo_batch.append({"stem": stem, "pose": build_pose(ex["comb"], ctx), "raw": raw})
                if args.apo:
                    apo_batch.append({"stem": stem, "pose": build_pose(ex["protein"], ctx), "raw": None})
            except Exception as exc:  # noqa: BLE001
                # any extraction/build failure -> passthrough input verbatim as _holo
                if raw is not None:
                    _atomic_write(out_dir / f"{stem}_holo.pdb", raw)
                    manifest.append((stem, f"passthrough_holo:{type(exc).__name__}", "", "", "", ""))
                else:
                    manifest.append((stem, f"error:{type(exc).__name__}", "", "", "", ""))
                print(f"[warn] {stem}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            flush(force=False)
        finally:
            if progress_queue is not None:
                progress_queue.put(1)
    flush(force=True)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    paths = read_list(args.list)

    stems = [Path(p).stem for p in paths]
    if len(set(stems)) != len(stems):
        print("[warn] duplicate input basenames detected; later outputs will overwrite earlier.",
              file=sys.stderr)

    ndev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_cuda = args.device.startswith("cuda") and ndev > 0
    nworkers = max(1, args.num_workers) if use_cuda else 1
    nworkers = min(nworkers, len(paths)) if paths else 1
    frozen = [k.replace("freeze_", "") for k, v in constraints_from_args(args).items() if v]
    print(f"complexes={len(paths)}  workers={nworkers}  batch_size={args.batch_size}  "
          f"apo={args.apo}  gpus_available={ndev}  "
          f"frozen={','.join(frozen) if frozen else 'none (all-atom)'}")

    # Parameterize the ligand ONCE (expensive) -> .tmol; workers load it cheaply.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmol_path = out_dir / f"_ligand_params_{args.res_name}.tmol"
    build_ligand_params(args.smiles, args.res_name, tmol_path)
    print(f"ligand parameterized -> {tmol_path}")

    import multiprocessing as mp
    from tqdm import tqdm
    ctxmp = mp.get_context("spawn")
    # Manager queue: a picklable proxy, so it survives being passed to spawned
    # workers via executor.submit(); workers put one item per finished complex.
    mgr = ctxmp.Manager()
    pq = mgr.Queue()

    def drain_bar(total, stop):
        """Run in a parent thread: pull completion ticks off pq into one tqdm bar."""
        with tqdm(total=total, unit="cplx", dynamic_ncols=True, smoothing=0.05) as bar:
            done = 0
            while done < total and not (stop.is_set() and pq.empty()):
                try:
                    pq.get(timeout=0.2)
                except queue.Empty:
                    continue
                done += 1
                bar.update(1)

    stop = threading.Event()
    bar_thread = threading.Thread(target=drain_bar, args=(len(paths), stop), daemon=True)
    bar_thread.start()

    if nworkers <= 1:
        device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        if device_str == "cuda":
            device_str = "cuda:0"
        manifest = run_shard(paths, args, device_str, tmol_path, progress_queue=pq)
    else:
        # multi-GPU sharding: round-robin shards (balances similar-sized proteins),
        # one worker per GPU, spawned (CUDA-safe) process pool.
        from concurrent.futures import ProcessPoolExecutor
        shards = [paths[i::nworkers] for i in range(nworkers)]
        manifest = []
        with ProcessPoolExecutor(max_workers=nworkers, mp_context=ctxmp) as ex:
            futs = [ex.submit(run_shard, shard, args, f"cuda:{i % ndev}", tmol_path, pq)
                    for i, shard in enumerate(shards)]
            for f in futs:
                manifest.extend(f.result())
    stop.set()
    bar_thread.join(timeout=2)
    mgr.shutdown()

    if args.manifest:
        import csv
        with open(args.manifest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stem", "status", "n_lig_H", "formal_charge", "e_before", "e_after"])
            w.writerows(manifest)
        print(f"wrote manifest: {args.manifest}")

    n_min = sum(1 for r in manifest if r[1].startswith("minimized_holo"))
    n_pass = sum(1 for r in manifest if r[1].startswith("passthrough_holo"))
    print(f"done: {len(paths)} inputs, holo minimized={n_min}, passthrough={n_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
