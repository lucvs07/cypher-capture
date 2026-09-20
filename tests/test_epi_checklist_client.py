"""
EpiChecklistClient: mesma escada de degradacao de CameraConfigClient (API ->
cache em disco -> None), mas para o checklist global de EPI da empresa
(achado de campo 2026-09-20).
"""
from datetime import datetime, timezone

from src.capture.epi_checklist_client import EpiChecklistClient

AGORA = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
CORPO = {"epis": ["HELMET", "VEST"]}


class TransporteFake:
    def __init__(self, status=200, corpo=None, erro=None):
        self.status = status
        self.corpo = corpo if corpo is not None else CORPO
        self.erro = erro
        self.urls: list[str] = []
        self.api_keys: list[object] = []

    def get_json(self, url, api_key=None):
        self.urls.append(url)
        self.api_keys.append(api_key)
        if self.erro is not None:
            raise self.erro
        return self.status, self.corpo


def cliente(tmp_path, transporte, **kwargs):
    return EpiChecklistClient(
        "http://api.local/api/v1",
        cache_path=str(tmp_path / "cache.json"),
        transport=transporte,
        now=lambda: AGORA,
        **kwargs,
    )


class TestBuscaNaApi:
    def test_monta_url_correta_e_envia_a_chave(self, tmp_path):
        transporte = TransporteFake()
        c = cliente(tmp_path, transporte)
        epis = c.fetch(api_key="minha-chave")
        assert transporte.urls == ["http://api.local/api/v1/config/epis"]
        assert transporte.api_keys == ["minha-chave"]
        assert epis == ["HELMET", "VEST"]

    def test_sem_api_key_nao_envia_header(self, tmp_path):
        transporte = TransporteFake()
        c = cliente(tmp_path, transporte)
        c.fetch()
        assert transporte.api_keys == [None]

    def test_lista_vazia_no_corpo_devolve_lista_vazia(self, tmp_path):
        transporte = TransporteFake(corpo={"epis": []})
        c = cliente(tmp_path, transporte)
        assert c.fetch() == []

    def test_corpo_sem_a_chave_epis_devolve_lista_vazia(self, tmp_path):
        transporte = TransporteFake(corpo={})
        c = cliente(tmp_path, transporte)
        assert c.fetch() == []


class TestDegradacaoParaCache:
    def test_falha_de_rede_cai_para_o_cache(self, tmp_path):
        transporte_ok = TransporteFake()
        c = cliente(tmp_path, transporte_ok)
        c.fetch()

        transporte_com_erro = TransporteFake(erro=ConnectionError("fora do ar"))
        c2 = cliente(tmp_path, transporte_com_erro)
        assert c2.fetch() == ["HELMET", "VEST"]

    def test_status_nao_200_cai_para_o_cache(self, tmp_path):
        transporte_ok = TransporteFake()
        c = cliente(tmp_path, transporte_ok)
        c.fetch()

        transporte_500 = TransporteFake(status=500, corpo={})
        c2 = cliente(tmp_path, transporte_500)
        assert c2.fetch() == ["HELMET", "VEST"]

    def test_sem_api_e_sem_cache_devolve_none(self, tmp_path):
        transporte = TransporteFake(erro=ConnectionError("fora do ar"))
        c = cliente(tmp_path, transporte)
        assert c.fetch() is None

    def test_cache_corrompido_e_ignorado_com_seguranca(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("{nao e json valido")
        transporte = TransporteFake(erro=ConnectionError("fora do ar"))
        c = cliente(tmp_path, transporte)
        assert c.fetch() is None
