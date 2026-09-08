#!/usr/bin/env python3
"""
CLI entry-point for the unified EPI + pose pipeline (Phase 2).

Drives UnifiedPipeline frame by frame so it can: save annotated video, run
headless, print a per-stage latency HUD, and surface events as they happen
(UnifiedPipeline.run_video discards events and has no output/HUD — this loop
supersedes it for live testing and measurement).

Usage:
    # webcam, headless (for measurement scripts on a GUI-less box)
    python src/run_pipeline.py --source 0 --no-display

    # video file, save annotated output, show window
    python src/run_pipeline.py --source data/validation/cenario1.mp4 \
        --output outputs/captures/cenario1_annot.mp4

    # RTSP stream with a custom zones file
    python src/run_pipeline.py --source rtsp://... --zones config/zones/zones.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.capture import (
    ApiKeyProvisioner, ApiKeyProvisioningError, CameraConfigClient,
    CameraDisabledError, CameraNotRegisteredError, CaptureConfig,
    MissingCredentialError, MissingGestorCredentialError, ResilientCapture,
    aplicar_opcoes_ffmpeg,
)
from src.fusion.pipeline import PipelineConfig, UnifiedPipeline
from src.notify import (
    ConsoleIncidentSink, HeartbeatSender, HttpIncidentSink, IncidentTracker,
    MissingApiKeyError, MjpegServer, NotifyConfig, TrackerConfig,
)
from src.utils.pose_visualization import draw_epi_boxes
from src.utils.video import apply_rotation
from src.zones import ZoneConfigClient, ZoneDetector, ZoneHotReloader, ZoneManager, Zone


def resolver_config_remota(cliente: CameraConfigClient, camera_id: str,
                            env, explicit_source: bool,
                            api_key: Optional[str] = None):
    """Busca a config remota; se falhar e houver --source, cede a ele em vez
    de derrubar o processo — --source existe para depurar exatamente essa
    situação (cadastro indisponivel, camera desativada, credencial ausente).
    """
    try:
        return cliente.fetch(camera_id, env, api_key=api_key)
    except (CameraDisabledError, CameraNotRegisteredError,
            MissingCredentialError) as erro:
        if explicit_source:
            print(f"[config] cadastro de {camera_id} indisponivel ({erro}); "
                  f"seguindo com --source")
            return None
        raise SystemExit(f"[config] {erro}")


def resolver_zonas(cliente: ZoneConfigClient, camera_id: str,
                   width: int, height: int,
                   api_key: Optional[str] = None) -> Optional[list[Zone]]:
    """Busca zonas remotas convertidas para a resolucao real do frame.

    None quando API e cache nao tem nada — o chamador mantem as zonas do
    arquivo local (config.zones_config) sem interrupcao (design doc de
    zonas de risco, §3.1, degrau 3). Diferente de resolver_config_remota,
    nao ha excecao de dominio a distinguir: zero zonas e sempre aceito.
    """
    return cliente.fetch(camera_id, width, height, api_key=api_key)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cypher — unified EPI + pose pipeline")
    p.add_argument("--source", default=None,
                   help="webcam index (e.g. 0) | video path | RTSP URL. "
                        "Opcional quando --camera-id é informado: nesse caso a "
                        "fonte vem do cadastro. Se os dois forem passados, "
                        "--source vence.")
    p.add_argument("--config", default="config/pose_config.yaml",
                   help="pose_config.yaml path (default: config/pose_config.yaml)")
    p.add_argument("--zones", default=None,
                   help="override zones JSON path (default: taken from config)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                   help="override inference device (default: taken from config; "
                        "use 'cuda' to meet the 200ms latency target)")
    p.add_argument("--output", default=None,
                   help="path to save annotated video (e.g. outputs/captures/run.mp4)")
    p.add_argument("--no-display", action="store_true",
                   help="run headless — no GUI window (for measurement / servers)")
    p.add_argument("--min-severity", type=int, default=2,
                   help="print events at/above this severity (0-4, default 2)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = until source ends / 'q')")
    p.add_argument("--show-epi", action="store_true",
                   help="dev: draw the raw EPI detection boxes from the PPE model")
    p.add_argument("--camera-id", default=None,
                   help="identificador desta camera (default: do config)")
    p.add_argument("--api-url", default=None,
                   help="base da API, ex. http://localhost:8080/api/v1 "
                        "(sem isso, os incidentes so sao impressos)")
    p.add_argument("--stream-port", type=int, default=None,
                   help="porta do stream MJPEG (default: do config)")
    p.add_argument("--no-stream", action="store_true",
                   help="nao sobe o servidor de video ao vivo")
    p.add_argument("--zone-refresh-seconds", type=float, default=30.0,
                   help="hot-reload: intervalo (em segundos) para rebuscar zonas "
                        "remotas em segundo plano e aplicar sem reiniciar o "
                        "pipeline; 0 desativa (default: 30). Ignorado quando "
                        "--zones (arquivo local explicito) e usado.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="graus (sentido horario) para corrigir uma camera "
                        "montada fisicamente virada (instalacao tipo corredor) "
                        "— aplicado em cada frame bruto antes de qualquer "
                        "analise. 90/270 trocam largura x altura, o que muda "
                        "a resolucao usada para buscar zonas no cadastro; "
                        "zonas calibradas na orientacao antiga precisam ser "
                        "re-desenhadas (scripts/annotate_zones.py) apos girar.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig.from_yaml(args.config)
    if args.zones:
        config.zones_config = args.zones
    if args.device:
        config.device = args.device

    print(f"[config] device={config.device}  pose={config.pose_model_path}  epi={config.epi_model_path}")
    print(f"[config] zones={config.zones_config}  target_fps={config.target_fps}  budget={config.max_latency_ms:.0f}ms")
    if args.rotate:
        print(f"[config] rotate={args.rotate}° (horario) aplicado em cada frame bruto")

    capture_config = CaptureConfig.from_yaml(args.config)
    aplicar_opcoes_ffmpeg(capture_config, os.environ)

    notify_config = NotifyConfig.from_yaml(args.config)
    if args.camera_id:
        notify_config.camera_id = args.camera_id
    if args.api_url:
        notify_config.api_url = args.api_url
    if args.stream_port is not None:
        notify_config.stream_port = args.stream_port
    if args.no_stream:
        notify_config.stream_enabled = False

    # Provisionamento antecipado, antes de qualquer busca de config/zonas
    # remotas: essas rotas exigem X-Cypher-Key ou sessao GESTOR (design doc
    # de zonas de risco §2, mesmo padrao ja usado por GET /cameras/{id}/config
    # desde E2) — sem a chave em maos ANTES da primeira chamada, a API sempre
    # responde 401 e a escada degrada pro proximo degrau mesmo quando uma
    # chave valida ja existe em cache (achado do teste manual ponta a ponta,
    # roteiro de entregas §5). So entra em jogo quando api_key_env NAO foi
    # configurado manualmente E ha credencial de gestor no ambiente — sem os
    # dois, o comportamento continua identico ao de antes (sem header nenhum).
    chave_provisionada = None
    if (args.camera_id and args.api_url and not notify_config.api_key_env
            and os.getenv("CYPHER_GESTOR_USER")):
        provisioner = ApiKeyProvisioner(notify_config.api_url)
        try:
            chave_provisionada = provisioner.obter(notify_config.camera_id, os.environ)
        except (ApiKeyProvisioningError, MissingGestorCredentialError) as erro:
            raise SystemExit(f"[notify] {erro}")
        print(f"[notify] X-Cypher-Key provisionada automaticamente para {notify_config.camera_id}")

    fonte = args.source
    origem_da_fonte = "--source"
    config_remota = None

    if args.camera_id and args.api_url:
        cliente = CameraConfigClient(args.api_url,
                                     cache_path=capture_config.cache_path)
        config_remota = resolver_config_remota(
            cliente, args.camera_id, os.environ, explicit_source=bool(args.source),
            api_key=chave_provisionada)

    if config_remota is not None:
        if args.source:
            print(f"[config] --source vence o cadastro de {args.camera_id} "
                  f"(depuracao contra arquivo)")
        else:
            fonte = config_remota.fonte
            origem_da_fonte = ("cadastro" if not config_remota.do_cache
                               else f"cache de {config_remota.idade_segundos / 3600:.1f}h")
    elif args.camera_id and not args.source:
        print(f"[config] cadastro indisponivel e sem cache para {args.camera_id}; "
              f"informe --source")

    if not fonte:
        p_erro = ("Informe --source, ou --camera-id com --api-url apontando "
                  "para uma camera cadastrada.")
        raise SystemExit(p_erro)

    print(f"[config] fonte={fonte} (origem: {origem_da_fonte})")

    pipeline = UnifiedPipeline(config)
    print(f"[zones]  {len(pipeline._zones.get_all())} zone(s) loaded")

    if (args.stream_port is None and config_remota is not None
            and not args.source and config_remota.stream_port):
        notify_config.stream_port = config_remota.stream_port

    if notify_config.enabled:
        uploader = None
        if os.getenv("CLOUDINARY_CLOUD_NAME"):
            from src.notify.sink import CloudinaryUploader
            uploader = CloudinaryUploader(folder=os.getenv("CLOUDINARY_FOLDER", "cypher-incidents"))
            print(f"[notify] snapshots serao enviados ao Cloudinary (pasta: {uploader.folder})")

        # chave_provisionada ja foi resolvida mais acima (antes da busca de
        # config/zonas remotas) — reaproveitada aqui, sem provisionar de novo.
        try:
            sink = HttpIncidentSink(
                notify_config.api_url,
                queue_size=notify_config.queue_size,
                spill_path=notify_config.spill_path,
                uploader=uploader,
                api_key_env=notify_config.api_key_env,
                api_key=chave_provisionada,
            )
            heartbeat_sender = HeartbeatSender(
                notify_config.api_url, notify_config.camera_id,
                api_key_env=notify_config.api_key_env, api_key=chave_provisionada,
                interval_seconds=notify_config.heartbeat_interval_seconds,
            )
        except MissingApiKeyError as erro:
            raise SystemExit(f"[notify] {erro}")
        print(f"[notify] enviando incidentes para {notify_config.api_url} "
              f"como {notify_config.camera_id} (run {notify_config.run_id})")
        if notify_config.api_key_env:
            print(f"[notify] X-Cypher-Key resolvido via {notify_config.api_key_env}")
    else:
        sink = ConsoleIncidentSink()
        print("[notify] sem api_url: incidentes apenas impressos")
        heartbeat_sender = None
    sink.start()

    tracker = IncidentTracker(
        TrackerConfig(
            camera_id=notify_config.camera_id,
            run_id=notify_config.run_id,
            min_severity=notify_config.min_severity,
            merge_window_seconds=notify_config.merge_window_seconds,
            evict_after_frames=notify_config.evict_after_frames,
            min_track_frames=notify_config.min_track_frames,
            fps=config.target_fps,
        ),
        on_open=sink.open,
        on_close=sink.close,
    )

    stream = None
    if notify_config.stream_enabled:
        stream = MjpegServer(port=notify_config.stream_port)
        stream.start()

        def _publish(frame):
            # pipeline.process_frame() publishes before returning, i.e.
            # before this module's own --show-epi overlay runs (below) —
            # draw it here too so the live stream carries the boxes, not
            # just the local window/--output file.
            if args.show_epi:
                draw_epi_boxes(frame, pipeline._last_epi_bboxes, pipeline._last_epi_labels)
            stream.publish(frame)

        pipeline.set_frame_publisher(_publish)
        print(f"[stream] video ao vivo em http://127.0.0.1:{stream.port}/stream")

    cap = ResilientCapture(
        fonte,
        backoff_base=capture_config.backoff_base_seconds,
        backoff_max=capture_config.backoff_max_seconds,
    ).open()
    width = cap.width
    height = cap.height
    if args.rotate in (90, 270):
        width, height = height, width
    src_fps = cap.fps

    zone_reloader = None
    if args.camera_id and args.api_url:
        zone_cliente = ZoneConfigClient(args.api_url, cache_path=capture_config.zones_cache_path)
        zonas_remotas = resolver_zonas(zone_cliente, args.camera_id, width, height,
                                       api_key=chave_provisionada)
        if zonas_remotas is not None:
            if args.zones:
                print(f"[zones]  --zones vence o cadastro de {args.camera_id} "
                      f"(depuracao contra arquivo)")
            else:
                pipeline._zones = ZoneManager(config.zones_config, zones=zonas_remotas)
                pipeline._zone_detector = ZoneDetector(pipeline._zones, config.approach_buffer_px)
                print(f"[zones]  {len(zonas_remotas)} zone(s) resolved remotely for {width}x{height}")

        # Hot-reload (design doc de zonas de risco §3.4/§7): sem isso, uma
        # zona criada/editada na UI so valia no proximo boot. Roda em thread
        # separada pra nao travar o loop de frames com a chamada de rede do
        # ZoneConfigClient (timeout de 5s do transporte real). --zones
        # (arquivo local explicito) sempre vence e desativa o hot-reload,
        # mesmo espirito de "--source vence o cadastro" ja usado acima.
        if not args.zones and args.zone_refresh_seconds > 0:
            def _aplicar_zonas_recarregadas(novas_zonas: list[Zone]) -> None:
                pipeline._zones = ZoneManager(config.zones_config, zones=novas_zonas)
                pipeline._zone_detector = ZoneDetector(pipeline._zones, config.approach_buffer_px)
                print(f"\n[zones]  {len(novas_zonas)} zone(s) atualizadas em segundo "
                      f"plano (hot-reload, sem reiniciar o pipeline)")

            zone_reloader = ZoneHotReloader(
                zone_cliente, args.camera_id, width, height,
                on_zonas=_aplicar_zonas_recarregadas,
                interval_seconds=args.zone_refresh_seconds,
                api_key=chave_provisionada,
            )
            if zonas_remotas is not None:
                zone_reloader.marcar_zonas_atuais(zonas_remotas)
            zone_reloader.start()
            print(f"[zones]  hot-reload ativo (revalida a cada {args.zone_refresh_seconds:.0f}s)")

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (width, height))
        print(f"[output] writing annotated video to {args.output} ({width}x{height} @ {src_fps:.0f}fps)")

    window = "Cypher — EPI + Pose (q to quit)"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    roll: deque[float] = deque(maxlen=30)   # rolling window for live FPS
    all_lat: list[float] = []               # full-run latencies for final stats
    frame_idx = 0
    t_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if cap.state == "encerrado":
                    break
                # Câmera ao vivo sem frame novo (ou reconectando): o laço
                # continua girando em vez de encerrar a execução.
                time.sleep(0.005)
                continue

            if args.rotate:
                frame = apply_rotation(frame, args.rotate)

            t0 = time.perf_counter()
            annotated, events = pipeline.process_frame(frame)
            dt_ms = (time.perf_counter() - t0) * 1000

            if args.show_epi and stream is None:
                # Stream path already drew these in the frame_publisher above
                # (same array, mutated in place) — avoid drawing them twice.
                annotated = draw_epi_boxes(
                    annotated, pipeline._last_epi_bboxes, pipeline._last_epi_labels
                )

            roll.append(dt_ms)
            all_lat.append(dt_ms)
            frame_idx += 1

            tm = pipeline.last_timings
            avg = sum(roll) / len(roll)
            fps = 1000.0 / avg if avg > 0 else 0.0
            epi_tag = "" if tm.get("epi_ran") else "(skip)"
            hud = (f"f{frame_idx:5d}  {dt_ms:6.1f}ms (avg {avg:5.1f} | {fps:4.1f} FPS)  "
                   f"pose {tm.get('pose_ms', 0):5.1f} | epi {tm.get('epi_ms', 0):5.1f}{epi_tag} | "
                   f"analysis {tm.get('analysis_ms', 0):4.1f} | overlay {tm.get('overlay_ms', 0):4.1f} | "
                   f"stride {pipeline._epi_stride}")
            print(hud, end="\r", flush=True)

            # Event sink: surface anything at/above the severity threshold
            for ev in events:
                if ev.severity >= args.min_severity:
                    present = ",".join(ev.epi_present) or "-"
                    viol = ",".join(ev.epi_violations) or "-"
                    print(f"\n[EVENT] f{frame_idx} id#{ev.person_id} sev={ev.severity} "
                          f"posture={ev.posture} reba={ev.reba_score}(AL{ev.action_level}) "
                          f"zone={ev.zone_status}/{ev.zone_name or '-'} ppe=[{present}] viol={viol} :: {ev.reason}")

            tracker.process(events, datetime.now(timezone.utc).astimezone(),
                            frame=annotated)

            if heartbeat_sender is not None:
                heartbeat_sender.enviar_se_necessario()

            # Burn a compact HUD onto the saved/displayed frame too
            cv2.putText(annotated, f"{dt_ms:.0f}ms  {fps:.1f}FPS  stride={pipeline._epi_stride}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        # O flush é o que impede que cada execução deixe incidentes órfãos:
        # sem ele, cenario2.mp4 encerra com 2 FALL_RISK abertos para sempre.
        tracker.flush(datetime.now(timezone.utc).astimezone())
        if zone_reloader is not None:
            zone_reloader.stop()
        sink.stop()
        if stream is not None:
            stream.stop()
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    lat = np.array(all_lat) if all_lat else np.array([0.0])
    eff_fps = frame_idx / wall if wall > 0 else 0.0
    p95 = float(np.percentile(lat, 95))
    print(f"\n\n{'=' * 60}")
    print(f"  Frames         : {frame_idx}")
    print(f"  Wall clock     : {wall:.1f}s  ({eff_fps:.1f} FPS effective)")
    print(f"  E2E latency    : mean={lat.mean():.1f}ms  p50={np.percentile(lat, 50):.1f}ms  "
          f"p95={p95:.1f}ms  max={lat.max():.1f}ms")
    target = config.max_latency_ms
    print(f"  Latency target : {'PASS ✓' if p95 <= target else 'FAIL ✗'}  (p95 ≤ {target:.0f}ms)")
    if args.output:
        print(f"  Saved          : {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
