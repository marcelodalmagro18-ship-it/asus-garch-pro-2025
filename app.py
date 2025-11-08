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

# Login (mesmo de antes)
if "users" not in st.session_state: st.session_state.users = {}
if "logado" not in st.session_state: st.session_state.logado = None

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
    st.markdown("**100% automático como seu notebook — escolhe o MELHOR MODELO por AIC! <3**")

    # meus_ativos.txt
    uploaded = st.file_uploader("Carregar meus_ativos.txt", type="txt")
    if uploaded:
        string = uploaded.read().decode("utf-8")
        st.session_state.ativos = [x.strip() for x in string.split(",") if x.strip()]
        st.success(f"Carregados {len(st.session_state.ativos)} ativos!")

    if "ativos" not in st.session_state:
        st.session_state.ativos = ["PETR4.SA", "VALE3.SA", "6A=F", "6B=F", "6C=F", "6E=F", "6J=F", "MES=F"]

    col_at1, col_at2 = st.columns([3,1])
    with col_at1:
        ativos_selecionados = st.multiselect("Ativos", st.session_state.ativos, default=st.session_state.ativos[:7])
    with col_at2:
        novo = st.text_input("Novo ativo")
        if st.button("ADICIONAR"):
            if novo not in st.session_state.ativos:
                st.session_state.ativos.append(novo.upper())
                st.rerun()
        if st.button("EXCLUIR"):
            for a in ativos_selecionados:
                if a in st.session_state.ativos:
                    st.session_state.ativos.remove(a)
            st.rerun()

    st.download_button("Baixar meus_ativos.txt", ",".join(st.session_state.ativos), "meus_ativos.txt")

    distribuicao = st.selectbox("Distribuição", ["normal", "t"])
    inicio = st.date_input("Início", datetime.now() - timedelta(days=1462))
    fim = st.date_input("Fim", datetime.now())
    alarme_percent = st.slider("Alarme Vol Atual > Longo em %", 0, 100, 20)

    if st.button("EXECUTAR PRO 3.9.4 – ANÁLISE AUTOMÁTICA"):
        relatorio = f"GARCH ANALYZER PRO 3.9.4 – ANÁLISE COMPLETA + REGRAS POR TIPO DE ATIVO\n"
        relatorio += f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        relatorio += f"Período analisado: {inicio} → {fim}\n"
        relatorio += f"Dias corridos: {(fim - inicio).days} | Dias úteis: ≈ {((fim - inicio).days * 252 / 365):.0f}\n\n"
        relatorio += "RESULTADOS DOS MODELOS VENCEDORES + INTERPRETAÇÃO AUTOMÁTICA\n"
        relatorio += "="*160 + "\n"
        relatorio += "Ativo    Modelo           AIC      LB     Ω            α          β          γ          Status     Interpretação\n"
        relatorio += "="*160 + "\n"

        modelos_list = [
            ("GARCH(1,1)", 1, 0, 1, "Garch"),
            ("GARCH(1,2)", 1, 0, 2, "Garch"),
            ("GARCH(2,1)", 2, 0, 1, "Garch"),
            ("EGARCH(1,1)", 1, 0, 1, "EGarch"),
            ("EGARCH(1,2)", 1, 0, 2, "EGarch"),
            ("GJR-GARCH(1,1,1)", 1, 1, 1, "GJR")
        ]

        for ativo in ativos_selecionados:
            with st.spinner(f"Testando 6 modelos em {ativo}..."):
                try:
                    data = yf.download(ativo, start=inicio, end=fim, progress=False)
                    if data.empty:
                        st.error(f"{ativo} sem dados")
                        continue
                    ret = data["Close"].pct_change().dropna() * 100
                    scaled = ret * 10

                    melhor_aic = np.inf
                    melhor_res = None
                    melhor_modelo_nome = ""

                    for nome, p, o, q, vol in modelos_list:
                        try:
                            am = arch_model(scaled, p=p, o=o, q=q, vol=vol, dist=distribuicao)
                            res = am.fit(disp="off")
                            if res.aic < melhor_aic:
                                melhor_aic = res.aic
                                melhor_res = res
                                melhor_modelo_nome = nome
                        except:
                            pass

                    if melhor_res is None:
                        st.error(f"{ativo} nenhum modelo convergiu")
                        continue

                    res = melhor_res
                    omega = res.params.get("omega", 0)
                    alpha_sum = sum(res.params.get(f"alpha[{i}]", 0) for i in range(1, 10))
                    beta_sum = sum(res.params.get(f"beta[{i}]", 0) for i in range(1, 10))
                    gamma = res.params.get("gamma[1]", 0) / 2 if "GJR" in melhor_modelo_nome else res.params.get("gamma[1]", 0)
                    vol_long = np.sqrt((omega / (1 - alpha_sum - beta_sum - gamma)) * 252) / 10
                    vol_atual = np.sqrt(res.conditional_volatility.iloc[-1]) * np.sqrt(252) / 10
                    lb = acorr_ljungbox(res.resid, lags=10, return_df=True)["lb_pvalue"].iloc[-1]

                    status = "EXCELENTE" if lb > 0.05 else "ATENCAO"
                    if "F" in ativo: interpret = "VOL TÉCNICA (FUTUROS)"
                    elif alpha_sum < 0.07: interpret = "FOREX CLÁSSICO" if any(x in ativo for x in ["EUR","USD","GBP","JPY"]) else "ACAO MADURA"
                    elif alpha_sum > 0.15: interpret = "ACAO VOLÁTIL"
                    elif "EGARCH" in melhor_modelo_nome and omega < -0.5: interpret = "QUEDAS EXPLODEM VOL!"
                    elif beta_sum > 0.98: interpret = "VOL DURA MUITO"
                    else: interpret = "Estável"

                    relatorio += f"{ativo:<8} {melhor_modelo_nome:<16} {melhor_aic:8.1f} {lb:6.3f} {omega:10.6f} {alpha_sum:10.6f} {beta_sum:10.6f} {gamma:10.6f} {status:<10} {interpret}\n"

                    # Métricas + gráfico
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.metric("Vol Longo", f"{vol_long:.4%}")
                        st.metric("Vol Atual", f"{vol_atual:.4%}")
                        st.metric("Diferença", f"{(vol_atual/vol_long-1)*100:+.2f}%")
                        st.metric("Modelo Vencedor", melhor_modelo_nome)
                    with col2:
                        fig, ax = plt.subplots(figsize=(10,5))
                        vol_plot = np.sqrt(res.conditional_volatility.iloc[-200:]) * np.sqrt(252) / 10
                        vol_plot.plot(ax=ax)
                        ax.axhline(vol_long, color="red", linestyle="--")
                        ax.set_title(f"{melhor_modelo_nome} - {ativo}")
                        st.pyplot(fig)

                    if vol_atual > vol_long * (1 + alarme_percent/100):
                        st.error(f"ALARME {ativo}: Vol Atual +{alarme_percent}% acima!")

                    # CSV MT5 vencedor
                    csv_mt5 = io.StringIO()
                    csv_mt5.write("Parametro,Valor\n")
                    csv_mt5.write(f"omega,{omega}\n")
                    for i in range(1,10):
                        a = res.params.get(f"alpha[{i}]",0)
                        b = res.params.get(f"beta[{i}]",0)
                        if a > 0: csv_mt5.write(f"alpha{i},{a}\n")
                        if b > 0: csv_mt5.write(f"beta{i},{b}\n")
                    g = res.params.get("gamma[1]",0)
                    if g > 0: csv_mt5.write(f"gamma1,{g}\n")
                    st.download_button(f"MT5 {ativo}", csv_mt5.getvalue(), f"{ativo}_MT5.csv", "text/csv")

                except Exception as e:
                    st.error(f"Erro {ativo}: {e}")

        relatorio += "="*160 + "\n"
        relatorio += "LEGENDA DAS INTERPRETAÇÕES AUTOMÁTICAS (v3.9.4)\n"
        relatorio += "="*160 + "\n"
        relatorio += "FOREX CLÁSSICO     → FOREX + GARCH + α<0.07 + β>0.90\n"
        relatorio += "VOL TÉCNICA        → FUTUROS + GARCH + α>0.08\n"
        relatorio += "ACAO MADURA        → AÇÃO + GARCH + α<0.07\n"
        relatorio += "ACAO VOLÁTIL       → AÇÃO + GARCH + α>0.15\n"
        relatorio += "QUEDAS EXPLODEM VOL! → EGARCH + Ω < -0.5\n"
        relatorio += "VOL DURA MUITO     → β > 0.98\n"

        st.code(relatorio, language="text")
        st.download_button("BAIXAR RELATÓRIO COMPLETO", relatorio, f"ANALISE_GARCH_PRO_{datetime.now().strftime('%Y-%m-%d')}.txt", "text/plain")
