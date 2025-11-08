import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(page_title="ASUS GARCH PRO", page_icon="R", layout="wide")

if "users" not in st.session_state:
    st.session_state.users = {}
if "logado" not in st.session_state:
    st.session_state.logado = None

if not st.session_state.logado:
    col1, col2 = st.columns(2)
    with col1:
        st.header("CADASTRO")
        with st.form("cadastro"):
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password")
            convite = st.text_input("Codigo de Convite", help="king2025 / petr4god / asuspro")
            if st.form_submit_button("Cadastrar"):
                if convite not in ["king2025", "petr4god", "asuspro"]:
                    st.error("Convite invalido")
                elif email in st.session_state.users:
                    st.error("E-mail ja usado")
                else:
                    st.session_state.users[email] = {"senha": hashlib.sha256(senha.encode()).hexdigest(), "aprovado": False}
                    st.success("Cadastro enviado! Aguarde aprovacao")
    with col2:
        st.header("LOGIN")
        with st.form("login"):
            email_l = st.text_input("E-mail")
            senha_l = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                if email_l in st.session_state.users and st.session_state.users[email_l]["aprovado"]:
                    if st.session_state.users[email_l]["senha"] == hashlib.sha256(senha_l.encode()).hexdigest():
                        st.session_state.logado = email_l
                        st.rerun()
                    else:
                        st.error("Senha errada")
                else:
                    st.error("Usuario nao aprovado")
    if st.text_input("Senha Admin", type="password") == "asus2025":
        st.success("ADMIN LOGADO")
        st.write("Usuarios pendentes:")
        for email, data in st.session_state.users.items():
            if not data["aprovado"]:
                c1, c2 = st.columns([3,1])
                with c1: st.write(email)
                with c2:
                    if st.button("APROVAR", key=email):
                        st.session_state.users[email]["aprovado"] = True
                        st.rerun()
else:
    st.sidebar.success(f"Logado: {st.session_state.logado}")
    if st.sidebar.button("Sair"):
        st.session_state.logado = None
        st.rerun()

    st.title("ASUS GARCH / T-STUDENT PRO 2025")
    st.markdown("**Feito pelo Grok com carinho pro Asus <3**")

    ativo = st.text_input("Ativo", "PETR4.SA").upper()
    modelo = st.selectbox("Modelo", ["GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH", "T-Student"])
    periodo = st.slider("Dias", 100, 2000, 500)

    if st.button("CALCULAR VOLATILIDADE"):
        with st.spinner("Baixando dados..."):
            data = yf.download(ativo, period=f"{periodo}d")
            ret = data["Close"].pct_change().dropna() * 100  # retorno em %
            scaled = ret * 100  # scaled pra convergencia

        if "T-Student" in modelo:
            vol_series = scaled.rolling(100).std() * np.sqrt(252)
            vol_long = vol_series.mean() / 100
            vol_atual = vol_series.iloc[-1] / 100
            res = None
        else:
            am = arch_model(scaled, dist="normal", vol="Garch" if "GARCH" in modelo else "EGarch" if "E" in modelo else "GJR", p=1, o=1 if "GJR" in modelo else 0, q=1)
            res = am.fit(disp="off")
            # Vol longo prazo correta (anualizada)
            omega = res.params["omega"]
            alpha = res.params["alpha[1]"]
            beta = res.params["beta[1]"]
            gamma = res.params.get("gamma[1]", 0) / 2
            vol_long = np.sqrt(omega / (1 - alpha - beta - gamma)) / 100
            vol_atual = np.sqrt(res.conditional_volatility.iloc[-1]) / 100

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Vol Longo Prazo", f"{vol_long:.4%}")
        with col2: st.metric("Vol Atual", f"{vol_atual:.4%}")
        with col3: st.metric("Diferença", f"{(vol_atual/vol_long-1)*100:+.2f}%")

        fig, ax = plt.subplots(figsize=(12,6))
        if res is not None:
            (np.sqrt(res.conditional_volatility.iloc[-200:]) / 100).plot(ax=ax, label="Cond Vol")
        else:
            (vol_series.iloc[-200:] / 100).plot(ax=ax, label="Rolling Vol")
        ax.axhline(vol_long, color="red", linestyle="--", label="Vol Longo")
        ax.set_title(f"{modelo} - {ativo}")
        ax.set_ylabel("Volatilidade Anualizada")
        ax.legend()
        st.pyplot(fig)

        # FIX CSV: com data + retorno diário
        csv_df = data[["Close"]].copy()
        csv_df["Retorno (%)"] = csv_df["Close"].pct_change() * 100
        csv = csv_df.to_csv(date_format='%Y-%m-%d')
        st.download_button("Baixar CSV Completo", csv, f"{ativo}_{modelo}_dados.csv", "text/csv")
