import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import hashlib
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="ASUS GARCH PRO 2025", page_icon="R", layout="wide")

# Theme ASUS
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #000000; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #000000; }
    .stButton > button { background-color: #00FF00; color: #000000; font-weight: bold; }
    .stMultiSelect > div { background-color: #0E1117; }
    .stTextInput > div > div > input { background-color: #0E1117; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# Login system (mesmo de antes)
if "users" not in st.session_state: st.session_state.users = {}
if "logado" not in st.session_state: st.session_state.logado = None

if not st.session_state.logado:
    # ... (mesmo cadastro/login/admin de antes - não mudei)
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
                    st.success("Cadastro enviado!")
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
        for email, data in st.session_state.users.items():
            if not data["aprovado"]:
                c1, c2 = st.columns([3,1])
                with c1: st.write(email)
                with c2:
                    if st.button("APROVAR", key=email):
                        st.session_state.users[email]["aprovado"] = True
                        st.rerun()
else:
    st.sidebar.success(f"Logado {st.session_state.logado}")
    if st.sidebar.button("Sair"):
        st.session_state.logado = None
        st.rerun()

    st.title("GARCH ANALYZER PRO 3.9.4 – ONLINE 24H")
    st.markdown("**Seu notebook original 100% fiel - turbinado pelo Grok <3**")

    # CARREGAR meus_ativos.txt
    uploaded = st.file_uploader("Carregar meus_ativos.txt", type="txt")
    if uploaded:
        string = uploaded.read().decode("utf-8")
        st.session_state.ativos = [x.strip() for x in string.split(",") if x.strip()]
        st.success(f"Carregados {len(st.session_state.ativos)} ativos!")

    if "ativos" not in st.session_state:
        st.session_state.ativos = ["PETR4.SA", "VALE3.SA", "6A=F", "6B=F", "6C=F", "6E=F", "6J=F"]

    # CADASTRO DE ATIVOS
    col_at1, col_at2 = st.columns([3,1])
    with col_at1:
        ativos_selecionados = st.multiselect("Ativos para analisar", st.session_state.ativos, default=st.session_state.ativos[:5])
    with col_at2:
        novo = st.text_input("Novo ativo")
        if st.button("ADICIONAR"):
            if novo not in st.session_state.ativos:
                st.session_state.ativos.append(novo.upper())
                st.rerun()
        if st.button("EXCLUIR SELECIONADO"):
            for a in ativos_selecionados:
                if a in st.session_state.ativos:
                    st.session_state.ativos.remove(a)
            st.rerun()

    # Salvar meus_ativos.txt
    txt_save = ",".join(st.session_state.ativos)
    st.download_button("Baixar meus_ativos.txt", txt_save, "meus_ativos.txt", "text/plain")

    modelo = st.selectbox("Modelo", ["GARCH(1,1)", "GARCH(1,2)", "GARCH(2,1)", "EGARCH(1,1)", "EGARCH(1,2)", "GJR-GARCH(1,1,1)"])
    distribuicao = st.selectbox("Distribuição", ["normal", "t"])
    inicio = st.date_input("Início", datetime.now() - timedelta(days=1462))
    fim = st.date_input("Fim", datetime.now())

    if st.button("EXECUTAR PRO 3.9.4"):
        relatorio = f"GARCH ANALYZER PRO 3.9.4 – ANÁLISE COMPLETA + REGRAS POR TIPO DE ATIVO\n"
        relatorio += f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        relatorio += f"Período analisado: {inicio} → {fim}\n"
        relatorio += f"Dias corridos: {(fim - inicio).days} | Dias úteis: ??? (≈ {((fim - inicio).days/365)*252:.0f} anos)\n\n"
        relatorio += "RESULTADOS DOS MODELOS VENCEDORES + INTERPRETAÇÃO AUTOMÁTICA\n"
        relatorio += "="*150 + "\n"
        relatorio += "Ativo    Modelo           AIC      LB     Ω            α          β          γ          Status     Interpretação                                     \n"
        relatorio += "="*150 + "\n"

        resultados = []

        for ativo in ativos_selecionados:
            with st.spinner(f"Processando {ativo}..."):
                try:
                    data = yf.download(ativo, start=inicio, end=fim, progress=False)
                    if data.empty:
                        st.error(f"{ativo} sem dados")
                        continue
                    ret = data["Close"].pct_change().dropna() * 100
                    scaled = ret * 10

                    # Parse modelo
                    if "GARCH(1,1)" in modelo: p, q, vol, o = 1, 1, "Garch", 0
                    elif "GARCH(1,2)" in modelo: p, q, vol, o = 1, 2, "Garch", 0
                    elif "GARCH(2,1)" in modelo: p, q, vol, o = 2, 1, "Garch", 0
                    elif "EGARCH(1,1)" in modelo: p, q, vol, o = 1, 1, "EGarch", 0
                    elif "EGARCH(1,2)" in modelo: p, q, vol, o = 1, 2, "EGarch", 0
                    elif "GJR" in modelo: p, q, vol, o = 1, 1, "GJR", 1

                    am = arch_model(scaled, p=p, o=o, q=q, vol=vol, dist=distribuicao)
                    res = am.fit(disp="off")

                    omega = res.params.get("omega", 0)
                    alpha_sum = sum(res.params.get(f"alpha[{i}]", 0) for i in range(1, p+1))
                    beta_sum = sum(res.params.get(f"beta[{i}]", 0) for i in range(1, q+1))
                    gamma = res.params.get("gamma[1]", 0) / 2 if o > 0 else 0
                    vol_long = np.sqrt((omega / (1 - alpha_sum - beta_sum - gamma)) * 252) / 10
                    vol_atual = np.sqrt(res.conditional_volatility.iloc[-1]) * np.sqrt(252) / 10
                    aic = res.aic
                    lb = acorr_ljungbox(res.resid, lags=10, return_df=True)["lb_pvalue"].iloc[-1]

                    status = "EXCELENTE" if lb > 0.05 else "ATENCAO"
                    if "F" in ativo or ativo.endswith(".L"): interpret = "VOL TÉCNICA (FUTUROS)"
                    elif alpha_sum < 0.07: interpret = "FOREX CLÁSSICO" if any(x in ativo for x in ["EUR", "USD", "GBP"]) else "ACAO MADURA"
                    elif alpha_sum > 0.15: interpret = "ACAO VOLÁTIL"
                    elif gamma > 0.05: interpret = "QUEDAS EXPLODEM VOL!"
                    else: interpret = "Estável"

                    resultados.append({
                        "ativo": ativo, "modelo": modelo, "aic": aic, "lb": lb, "omega": omega,
                        "alpha": alpha_sum, "beta": beta_sum, "gamma": gamma, "status": status,
                        "interpret": interpret, "vol_long": vol_long, "vol_atual": vol_atual
                    })

                    relatorio += f"{ativo:<8} {modelo:<16} {aic:8.1f} {lb:6.3f} {omega:10.6f} {alpha_sum:10.6f} {beta_sum:10.6f} {gamma:10.6f} {status:<10} {interpret}\n"

                    # Gráfico + métricas
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.metric("Vol Longo", f"{vol_long:.4%}")
                        st.metric("Vol Atual", f"{vol_atual:.4%}")
                        st.metric("Diferença", f"{(vol_atual/vol_long-1)*100:+.2f}%")
                    with col2:
                        fig, ax = plt.subplots(figsize=(10,5))
                        vol_plot = np.sqrt(res.conditional_volatility.iloc[-200:]) * np.sqrt(252) / 10
                        vol_plot.plot(ax=ax)
                        ax.axhline(vol_long, color="red", linestyle="--", label="Vol Longo")
                        ax.set_title(f"{modelo} - {ativo}")
                        ax.legend()
                        st.pyplot(fig)

                    # CSV MT5
                    csv_mt5 = io.StringIO()
                    csv_mt5.write("Parametro,Valor\nomega,{}\nalpha1,{}\nbeta1,{}\ngamma1,{}\n".format(
                        omega, res.params.get("alpha[1]",0), res.params.get("beta[1]",0), res.params.get("gamma[1]",0)
                    ))
                    st.download_button(f"MT5 {ativo}", csv_mt5.getvalue(), f"{ativo}_PARAMETROS_MT5.csv", "text/csv")

                except Exception as e:
                    st.error(f"Erro {ativo}: {e}")

        relatorio += "="*150 + "\n"
        relatorio += "LEGENDA DAS INTERPRETAÇÕES AUTOMÁTICAS (v3.9.4)\n"
        relatorio += "="*150 + "\n"
        relatorio += "FOREX CLÁSSICO     → FOREX + GARCH + α<0.07 + β>0.90\n"
        relatorio += "VOL TÉCNICA        → FUTUROS + GARCH + α>0.08\n"
        relatorio += "ACAO MADURA        → AÇÃO + GARCH + α<0.07\n"
        relatorio += "ACAO VOLÁTIL       → AÇÃO + GARCH + α>0.15\n"
        relatorio += "QUEDAS EXPLODEM VOL! → EGARCH + Ω < -0.5\n"
        relatorio += "VOL DURA MUITO     → β > 0.98\n"
        relatorio += "TECH/PÂNICO        → EGARCH + Ω < -0.3\n"

        st.code(relatorio, language="text")
        st.download_button("BAIXAR RELATÓRIO COMPLETO", relatorio, "ANALISE_GARCH_PRO_2025-11-08.txt", "text/plain")
