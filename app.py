import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import hashlib
import io
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime

st.set_page_config(page_title="ASUS GARCH PRO 2025", page_icon="R", layout="wide")

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
                    st.success("Cadastro enviado Aguarde aprovacao")
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
        st.write("Usuarios pendentes")
        for email, data in st.session_state.users.items():
            if not data["aprovado"]:
                c1, c2 = st.columns([3,1])
                with c1: st.write(email)
                with c2:
                    if st.button("APROVAR", key=email):
                        st.session_state.users[email]["aprovado"] = True
                        st.rerun()
else:
    st.sidebar.success("Logado " + st.session_state.logado)
    if st.sidebar.button("Sair"):
        st.session_state.logado = None
        st.rerun()

    st.title("ASUS GARCH ANALYZER PRO 3.9.4 ONLINE")
    st.markdown("Seu codigo original turbinado pelo Grok")

    ativo = st.text_input("Ativo", "PETR4.SA").upper()
    modelo_str = st.selectbox("Modelo", ["GARCH(1,1)", "GARCH(1,2)", "GARCH(2,1)", "EGARCH(1,1)", "EGARCH(1,2)", "GJR-GARCH(1,1,1)", "T-Student"])
    distribuicao = st.selectbox("Distribuicao", ["normal", "t"])
    periodo = st.slider("Dias", 100, 2000, 500)
    alarme_percent = st.slider("Alarme se Vol Atual > Vol Longo em %", 0, 100, 20)

    if st.button("EXECUTAR PRO 3.9.4"):
        with st.spinner("Baixando dados e ajustando modelo"):
            data = yf.download(ativo, period=f"{periodo}d", progress=False)
            if data.empty:
                st.error("Ativo nao encontrado ou sem dados")
                st.stop()
            ret = data["Close"].pct_change().dropna() * 100
            scaled = ret * 10

            if "T-Student" in modelo_str:
                vol_series = scaled.rolling(100).std() * np.sqrt(252)
                vol_long = vol_series.mean() / 10
                vol_atual = vol_series.iloc[-1] / 10
                res = None
            else:
                if "GARCH" in modelo_str and "E" not in modelo_str and "GJR" not in modelo_str:
                    p = int(modelo_str.split("(")[1].split(",")[0])
                    q = int(modelo_str.split(",")[1].split(")")[0])
                    vol_type = "Garch"
                    o = 0
                elif "EGARCH" in modelo_str:
                    p = int(modelo_str.split("(")[1].split(",")[0])
                    q = int(modelo_str.split(",")[1].split(")")[0])
                    vol_type = "EGarch"
                    o = 0
                elif "GJR" in modelo_str:
                    p, o, q = 1, 1, 1
                    vol_type = "GJR"
                else:
                    p, o, q = 1, 0, 1
                    vol_type = "Garch"

                am = arch_model(scaled, p=p, o=o, q=q, vol=vol_type, dist=distribuicao)
                res = am.fit(disp="off")

                omega = res.params["omega"]
                alpha_sum = sum(res.params.get(f"alpha[{i}]", 0) for i in range(1, p+1))
                beta_sum = sum(res.params.get(f"beta[{i}]", 0) for i in range(1, q+1))
                gamma = res.params.get("gamma[1]", 0) / 2 if "GJR" in modelo_str else 0
                vol_long = np.sqrt((omega / (1 - alpha_sum - beta_sum - gamma)) * 252) / 10
                vol_atual = np.sqrt(res.conditional_volatility.iloc[-1]) * np.sqrt(252) / 10

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Vol Longo Prazo", f"{vol_long:.4%}")
            with col2: st.metric("Vol Atual", f"{vol_atual:.4%}")
            with col3: st.metric("Diferenca", f"{(vol_atual/vol_long-1)*100:+.2f}%")

            fig, ax = plt.subplots(figsize=(12,6))
            if res is not None:
                vol_plot = np.sqrt(res.conditional_volatility.iloc[-200:]) * np.sqrt(252) / 10
                vol_plot.plot(ax=ax, label="Cond Vol")
            else:
                vol_plot = vol_series.iloc[-200:] / 10
                vol_plot.plot(ax=ax, label="Rolling Vol")
            ax.axhline(vol_long, color="red", linestyle="--", label="Vol Longo")
            ax.set_title(modelo_str + " - " + ativo)
            ax.set_ylabel("Vol Anualizada (%)")
            ax.legend()
            st.pyplot(fig)

            if vol_atual > vol_long * (1 + alarme_percent/100):
                st.error("ALTO RISCO: Vol Atual acima do limite")

            relatorio = "GARCH ANALYZER PRO 3.9.4 - " + ativo + " - " + datetime.now().strftime("%Y-%m-%d") + "\n\n"
            relatorio += "Modelo: " + modelo_str + " | Dist: " + distribuicao + "\n"
            if res is not None:
                relatorio += "Omega: " + str(res.params.get("omega", 0)) + "\n"
                relatorio += "Alpha total: " + str(alpha_sum) + "\n"
                relatorio += "Beta total: " + str(beta_sum) + "\n"
                relatorio += "Gamma/2: " + str(gamma) + "\n"
                relatorio += "AIC: " + str(res.aic) + "\n"
            relatorio += "Vol Longo: " + f"{vol_long:.4%}" + "\n"
            relatorio += "Vol Atual: " + f"{vol_atual:.4%}" + "\n"
            st.code(relatorio)

            csv_mt5 = "Parametro,Valor\nomega," + str(res.params.get("omega", 0)) + "\nalpha1," + str(res.params.get("alpha[1]", 0)) + "\nbeta1," + str(res.params.get("beta[1]", 0)) + "\ngamma1," + str(res.params.get("gamma[1]", 0)) + "\n"
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("Baixar Relatorio TXT", relatorio, ativo + "_relatorio.txt", "text/plain")
            with col_d2:
                st.download_button("Baixar PARAMETROS_MT5.csv", csv_mt5, ativo + "_PARAMETROS_MT5.csv", "text/csv")

            csv_full = data.copy()
            csv_full["Retorno (%)"] = ret
            csv_data = csv_full.to_csv()
            st.download_button("Baixar Dados Completos", csv_data, ativo + "_dados.csv", "text/csv")
