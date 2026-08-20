"""
De onde o pipeline tira sua fonte de captura.

O cadastro guarda a fonte com placeholders — ``rtsp://${U}:${P}@host/stream``
— e o segredo vive no ambiente do host que abre a câmera. Assim a senha nunca
atravessa a rede, nunca entra no banco e nunca aparece em backup nem em log
de requisição.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional

PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MissingCredentialError(Exception):
    """Uma variável referenciada pela fonte não existe no ambiente."""

    def __init__(self, variaveis: list[str]) -> None:
        self.variaveis = variaveis
        super().__init__(
            "Variaveis de ambiente ausentes para expandir a fonte da camera: "
            + ", ".join(variaveis)
            + ". Defina-as no .env do host (ou exporte antes de subir o pipeline)."
        )


def expand_placeholders(texto: Optional[str], env: Mapping[str, str]) -> Optional[str]:
    """Substitui ``${VAR}`` pelos valores de ``env``.

    Falha nomeando **todas** as variáveis ausentes: descobrir uma por execução
    é hostil quando faltam usuário e senha. Valor vazio conta como ausente,
    porque produziria uma URL sintaticamente válida e um erro de autenticação
    obscuro na câmera.
    """
    if not texto:
        return texto

    faltando: list[str] = []
    for nome in PLACEHOLDER.findall(texto):
        if not env.get(nome) and nome not in faltando:
            faltando.append(nome)
    if faltando:
        raise MissingCredentialError(faltando)

    return PLACEHOLDER.sub(lambda m: env[m.group(1)], texto)
