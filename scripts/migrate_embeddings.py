"""Re-extract all enrollment embeddings with the current encoder and rebuild fused voiceprints.

Run this after switching the encoder (e.g. ECAPA -> ERes2NetV2). Embeddings from different
models live in different vector spaces, so existing stored embeddings must be regenerated
from the saved enrollment audio before 1:1/1:N verification will work again.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_DIR
from database import DatabaseManager
from voiceprint import VoiceprintExtractor


def _resolve_audio(ap, uid):
    if ap and os.path.exists(ap):
        return ap
    if not ap:
        return None
    cand = os.path.join(AUDIO_DIR, "enrollment", str(uid), os.path.basename(ap))
    return cand if os.path.exists(cand) else None


def main():
    db = DatabaseManager()
    vpe = VoiceprintExtractor()
    print(f"Encoder: {vpe.encoder_name} ({vpe.model_name})")
    users = db.get_all_users()
    total_seg = 0
    migrated_users = 0
    for u in users:
        uid = u["id"]
        segs = db.get_enrollment_segments(uid)
        if not segs:
            continue
        new_embs = []
        for seg in segs:
            ap = _resolve_audio(seg["audio_path"], uid)
            if not ap:
                print(f"  [skip] user {uid} seg {seg['id']}: audio missing ({seg['audio_path']})")
                continue
            emb = vpe.extract_embedding(ap)
            conn = db.get_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE enrollment_segments SET embedding=?, audio_path=? WHERE id=?",
                (emb.tobytes(), ap, seg["id"]),
            )
            conn.commit()
            conn.close()
            new_embs.append(emb)
            total_seg += 1
        if new_embs:
            fused = vpe._fuse_embeddings(new_embs)
            db.replace_voiceprint(uid, fused, sample_count=len(new_embs))
            migrated_users += 1
            print(f"  user {uid} {u['name']}: {len(new_embs)} segments re-extracted -> fused")
    print(f"Migration complete: {total_seg} segments, {migrated_users} users.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
