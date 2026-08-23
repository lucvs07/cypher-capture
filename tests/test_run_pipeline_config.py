"""
Testes de ``resolver_config_remota`` (src/run_pipeline.py).

O que está sendo protegido: quando o cadastro devolve uma resposta
definitiva (câmera desativada, não cadastrada, ou credencial ausente no
ambiente deste host), ``--source`` precisa poder vencer em vez de derrubar o
processo com um traceback — essa é exatamente a situação em que um operador
usa ``--source`` para depurar contra um arquivo. Sem ``--source`` disponível,
o processo deve encerrar com uma mensagem específica em vez de seguir sem
fonte nenhuma.

Não é praticável testar isso via ``main()`` (argparse, pipeline completo,
etc.), então o helper é testado diretamente com um cliente falso — nenhuma
rede, câmera ou GPU envolvida.
"""
from __future__ import annotations

import pytest

from src.capture import CameraConfig
from src.capture.config_client import (
    CameraDisabledError,
    CameraNotRegisteredError,
    MissingCredentialError,
)
from src.run_pipeline import resolver_config_remota


class ClienteFake:
    """Cliente de config com .fetch() controlável: levanta o que for pedido,
    ou devolve uma CameraConfig normal."""

    def __init__(self, erro=None, config=None):
        self.erro = erro
        self.config = config

    def fetch(self, camera_id, env):
        if self.erro is not None:
            raise self.erro
        return self.config


ERROS_DEFINITIVOS = [
    CameraDisabledError("Camera desativada no cadastro: CAM-X"),
    CameraNotRegisteredError("Camera nao cadastrada: CAM-X"),
    MissingCredentialError(["CAM_USER"]),
]


@pytest.mark.parametrize("erro", ERROS_DEFINITIVOS)
def test_com_source_explicito_cede_a_ele_em_vez_de_derrubar(erro):
    cliente = ClienteFake(erro=erro)

    resultado = resolver_config_remota(cliente, "CAM-X", {}, explicit_source=True)

    assert resultado is None


@pytest.mark.parametrize("erro", ERROS_DEFINITIVOS)
def test_sem_source_explicito_encerra_o_processo(erro):
    cliente = ClienteFake(erro=erro)

    with pytest.raises(SystemExit):
        resolver_config_remota(cliente, "CAM-X", {}, explicit_source=False)


@pytest.mark.parametrize("explicit_source", [True, False])
def test_config_normal_e_devolvida_sem_alteracao(explicit_source):
    config = CameraConfig(
        codigo="CAM-X", nome="Esteira", estacao_id="EST-01",
        fonte="rtsp://10.0.0.7/stream", stream_port=8090, ativa=True,
        origem="api",
    )
    cliente = ClienteFake(config=config)

    resultado = resolver_config_remota(cliente, "CAM-X", {},
                                       explicit_source=explicit_source)

    assert resultado is config
