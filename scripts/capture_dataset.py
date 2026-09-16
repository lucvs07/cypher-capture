#!/usr/bin/env python3
"""
Modo teste: captura frames crus (sem overlay) das cameras RTSP cadastradas
e sobe direto pro Cloudinary, pra formar um dataset de retreino do
detector de EPI.

Nao passa pelo pipeline de pose/EPI — e so captura + upload, entao roda em
paralelo com run_pipeline.py sem disputar GPU (mas disputa a conexao RTSP:
algumas cameras baratas limitam quantos clientes RTSP conectam ao mesmo
tempo no mesmo canal; se a captura falhar com o pipeline de deteccao
rodando na mesma camera, tente rodar so um dos dois primeiro).

Usage:
    python scripts/capture_dataset.py --api-url http://localhost:8080/api/v1
    python scripts/capture_dataset.py --cameras CAM-FRESA1,CAM-SICK \
        --interval-seconds 5 --duration-seconds 600
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

from src.capture import (
    ApiKeyProvisioner, ApiKeyProvisioningError, CameraConfigClient,
    MissingCredentialError, MissingGestorCredentialError, ResilientCapture,
)

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

# Todas as cameras reais cadastradas na conta demo, exceto a webcam do
# notebook — essa e' pra teste manual do operador, nao serve pro dataset.
DEFAULT_CAMERAS = ["CAM-FRESA1", "CAM-FRESA2", "CAM-TENDA", "CAM-SALA4", "CAM-SICK"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", default="http://localhost:8080/api/v1",
                   help="base da API do cadastro (default: %(default)s)")
    p.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS),
                   help=f"lista separada por virgula (default: {','.join(DEFAULT_CAMERAS)})")
    p.add_argument("--interval-seconds", type=float, default=5.0,
                   help="intervalo entre capturas, por camera (default: 5s)")
    p.add_argument("--duration-seconds", type=float, default=600.0,
                   help="por quanto tempo capturar cada camera (default: 600s = 10min)")
    p.add_argument("--folder", default="cypher-capture/dataset",
                   help="pasta no Cloudinary (default: %(default)s)")
    return p.parse_args()


def capturar_camera(camera_id: str, api_url: str, interval_s: float,
                     duration_s: float, folder: str) -> None:
    api_key = None
    if os.getenv("CYPHER_GESTOR_USER"):
        try:
            api_key = ApiKeyProvisioner(api_url).obter(camera_id, os.environ)
        except (ApiKeyProvisioningError, MissingGestorCredentialError) as erro:
            print(f"[{camera_id}] {erro}")
            return

    cliente = CameraConfigClient(api_url, cache_path=f"outputs/dataset-cache-{camera_id}.json")
    try:
        config = cliente.fetch(camera_id, os.environ, api_key=api_key)
    except MissingCredentialError as erro:
        print(f"[{camera_id}] {erro}")
        return
    if config is None or not config.fonte:
        print(f"[{camera_id}] cadastro indisponivel ou sem fonte — pulando")
        return
    if not config.ativa:
        print(f"[{camera_id}] desativada no cadastro — pulando")
        return

    # Achado de campo (SPI 2026-09-16): manter a conexao RTSP aberta pela
    # duracao inteira derrubou o cliente do run_pipeline.py rodando ao mesmo
    # tempo na mesma camera — varias dessas cameras baratas so aceitam 1
    # cliente RTSP por canal, e o segundo cliente expulsa o primeiro. Agora
    # conecta, tira 1 frame, desconecta — a janela de disputa cai de
    # "duration_s inteiro" pra "menos de 1s por captura", nao elimina o
    # risco mas reduz bastante.
    print(f"[{camera_id}] capturando por {duration_s:.0f}s, 1 frame a cada {interval_s:.0f}s "
          f"(conecta so no instante da captura)")
    enviados = 0
    t_fim = time.monotonic() + duration_s

    while time.monotonic() < t_fim:
        inicio_ciclo = time.monotonic()

        try:
            cap = ResilientCapture(config.fonte).open()
        except RuntimeError as erro:
            print(f"[{camera_id}] falha ao conectar nesta captura: {erro}")
            time.sleep(max(0.0, interval_s - (time.monotonic() - inicio_ciclo)))
            continue

        try:
            # ResilientCapture.read() para fonte ao vivo eh nao-bloqueante:
            # devolve (False, None) se a thread de leitura em background
            # ainda nao decodificou o 1o frame. O handshake RTSP leva
            # tipicamente 1-3s, entao um read() unico logo apos open()
            # falha quase sempre — poll curto ate um frame chegar ou estourar
            # o timeout.
            ret, frame = False, None
            limite_leitura = time.monotonic() + 4.0
            while time.monotonic() < limite_leitura:
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
        finally:
            cap.release()

        if not ret:
            print(f"[{camera_id}] frame indisponivel nesta captura, tentando de novo em {interval_s:.0f}s")
            time.sleep(max(0.0, interval_s - (time.monotonic() - inicio_ciclo)))
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            resposta = cloudinary.uploader.upload(
                tmp.name, folder=f"{folder}/{camera_id}", public_id=ts)
        os.unlink(tmp.name)
        enviados += 1
        print(f"[{camera_id}] {enviados:4d} enviado: {resposta['secure_url']}")

        # Desconta do intervalo pedido o tempo ja gasto conectando, lendo e
        # subindo pro Cloudinary — senao a cadencia real fica mais espacada
        # que --interval-seconds (o usuario pediu "diversas fotos", nao
        # queremos entregar menos por causa do overhead da conexao nova a
        # cada ciclo).
        time.sleep(max(0.0, interval_s - (time.monotonic() - inicio_ciclo)))

    print(f"[{camera_id}] encerrado — {enviados} frame(s) enviados")


def main() -> None:
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    threads = [
        threading.Thread(
            target=capturar_camera,
            args=(cam, args.api_url, args.interval_seconds, args.duration_seconds, args.folder),
            name=cam,
        )
        for cam in cameras
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nTodas as cameras encerradas.")


if __name__ == "__main__":
    main()
