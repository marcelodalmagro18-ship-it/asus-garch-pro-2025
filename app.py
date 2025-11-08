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

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #000000; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #000000; }
    .stButton > button { background-color: #00FF00; color: #000000; font-weight: bold; }
    .stMultiSelect > div { background-color: #0E1117; }
    .stTextInput > div > div > input { background-color: #0E1117; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

if "users" not in st.session_state: st.session_state.users = {}
if "logado" not in st.session_state: st.session_state.logado = None

if not st.session_state.logado:
    col1, col2 = st.columns(2)
    with col1:
        st.header("CADASTRO")
        with st.form("cadastro"):
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password")
            convite = st.text_input("Convite", help="king2025 / petr4god / asuspro")
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
        with st.form("vip_form"):
            email_vip = st.text_input("E-mail VIP")
            senha_vip = st.text_input("Senha VIP", type="password")
            if st.form_submit_button("CADASTRAR + APROVAR"):
                if email_vip in st.session_state.users:
                    st.error("Ja existe")
                else:
                    st.session_state.users[email_vip] = {"senha": hashlib.sha256(senha_vip.encode()).hexdigest(), "aprovado": True}
                    st.success(f"{email_vip} LIBERADO!")
                    st.code(f"E-mail: {email_vip}\nSenha: {senha_vip}")
else:
    st.sidebar.success(f"Logado {st.session_state.logado}")
    if st.sidebar.button("Sair"):
        st.session_state.logado = None
        st.rerun()

    st.title("GARCH ANALYZER PRO 3.9.4 ONLINE 24H")
    st.markdown("**v16.1 FÓRMULAS 100% IGUAIS AO NOTEBOOK - γ CORRETO EGARCH full, GJR /2**")

    uploaded = st.file_uploader("Carregar meus_ativos.txt", type="txt")
    if uploaded:
        st.session_state.ativos = [x.strip() for x in uploaded.read().decode("utf-8").split(",") if x.strip()]
        st.success("Ativos carregados!")

    if "ativos" not in st.session_state:
        st.session_state.ativos = ["PETR4.SA", "VALE3.SA", "6A=F", "6B=F", "6C=F", "6E=F", "6J=F"]

    col_at1, col_at2 = st.columns([3,1])
    with col_at1:
        ativos_sel = st.multiselect("Ativos", st.session_state.ativos, default=st.session_state.ativos[:5])
    with col_at2:
        novo = st.text_input("Novo ativo")
        if st.button("ADICIONAR"):
            if novo.upper() not in st.session_state.ativos:
                st.session_state.ativos.append(novo.upper())
                st.rerun()
        if st.button("EXCLUIR"):
            for a in ativos_sel:
                if a in st.session_state.ativos:
                    st.session_state.ativos.remove(a)
            st.rerun()

    st.download_button("Baixar meus_ativos.txt", ",".join(st.session_state.ativos), "meus_ativos.txt")

    distribuicao = st.selectbox("Distribuicao", ["normal", "t"])
    inicio = st.date_input("Inicio", datetime.now() - timedelta(days=1462))
    fim = st.date_input("Fim", datetime.now())
    alarme_percent = st.slider("Alarme Vol > Longo em %", 0, 100, 20)

    if st.button("EXECUTAR PRO 3.9.4"):
        relatorio = "GARCH ANALYZER PRO 3.9.4 – ANÁLISE COMPLETA + REGRAS POR TIPO DE ATIVO\n"
        relatorio += f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        relatorio += f"Período analisado: {inicio} → {fim}\n"
        relatorio += f"Dias corridos: {timedelta(days=(fim - inicio).days)}\n\n"
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

        for ativo in ativos_sel:
            with st.spinner(f"Testando 6 modelos em {ativo}..."):
                try:
                    data = yf.download(ativo, start=inicio, end=fim, progress=False)
                    if data.empty:
                        st.error(f"{ativo} sem dados")
                        continue
                    ret = data["Close"].pct_change().dropna() * 100
                    scaled = ret * 10  # EXATO DO SEU NOTEBOOK

                    melhor_aic = np.inf
                    melhor_res = None
                    melhor_nome = ""
                    melhor_p = melhor_o = melhor_q = 1
                    melhor_vol = "Garch"

                    for nome, p, o, q, vol in modelos_list:
                        try:
                            am = arch_model(scaled, p=p, o=o, q=q, vol=vol, dist=distribuicao)
                            res = am.fit(disp="off")
                            if res.aic < melhor_aic:
                                melhor_aic = res.aic
                                melhor_res = res
                                melhor_nome = nome
                                melhor_p, melhor_o, melhor_q = p, o, q
                                melhor_vol = vol
                        except:
                            pass

                    if melhor_res is None:
                        st.error(f"{ativo} nenhum modelo convergiu")
                        continue

                    res = melhor_res
                    omega = res.params.get("omega", 0)
                    alpha_sum = sum(res.params.get(f"alpha[{i}]", 0) for i in range(1, melhor_p + 1))
                    beta_sum = sum(res.params.get(f"beta[{i}]", 0) for i in range(1, melhor_q + 1))

                    # GAMMA CORRETO POR MODELO (100% IGUAL SEU NOTEBOOK + DOC)
                    gamma_param = res.params.get("gamma[1]", 0.0)
                    if "EGARCH" in melhor_nome:
                        gamma = gamma_param  # EGARCH: full (não /2)
                        gamma_for_vol = gamma_param
                    elif "GJR" in melhor_nome:
                        gamma = gamma_param / 2  # GJR: mostra metade (efeito real)
                        gamma_for_vol = gamma_param / 2  # vol_long usa /2
                    else:
                        gamma = 0.0
                        gamma_for_vol = 0.0

                    # PERSISTENCE + VOL_LONG EXATO (linha por linha igual seu notebook)
                    persistence = alpha_sum + beta_sum + gamma_for_vol
                    if persistence >= 1.0:
                        vol_long = np.sqrt(omega / 0.0001) / 10
                    else:
                        vol_long = np.sqrt(omega / (1 - persistence)) / 10
                    vol_atual = np.sqrt(res.conditional_volatility.iloc[-1]) / 10
                    vol_long_anual = vol_long * np.sqrt(252)
                    vol_atual_anual = vol_atual * np.sqrt(252)
                    lb = acorr_ljungbox(res.resid, lags=10, return_df=True)["lb_pvalue"].iloc[-1]

                    status = "EXCELENTE" if lb > 0.05 else "ATENCAO"
                    interpret = "Estavel"
                    if "=F" in ativo: interpret = "VOL TECNICA (FUTUROS)"
                    elif alpha_sum < 0.07: interpret = "ACAO MADURA"
                    elif alpha_sum > 0.15: interpret = "ACAO VOLATIL"
                    elif "EGARCH" in melhor_nome and omega < -0.5: interpret = "QUEDAS EXPLODEM VOL!"
                    elif beta_sum > 0.98: interpret = "VOL DURA MUITO"

                    relatorio += f"{ativo:<8} {melhor_nome:<16} {melhor_aic:8.1f} {lb:6.3f} {omega:10.6f} {alpha_sum:10.6f} {beta_sum:10.6f} {gamma:10.6f} {status:<10} {interpret}\n"

                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.metric("Vol Longo (%)", f"{vol_long_anual:.4%}")
                        st.metric("Vol Atual (%)", f"{vol_atual_anual:.4%}")
                        st.metric("Diferenca", f"{(vol_atual_anual/vol_long_anual-1)*100:+.2f}%")
                        st.metric("Modelo Vencedor", melhor_nome)
                    with col2:
                        fig, ax = plt.subplots(figsize=(10,5))
                        vol_plot = np.sqrt(res.conditional_volatility.iloc[-200:]) / 10 * np.sqrt(252)
                        vol_plot.plot(ax=ax)
                        ax.axhline(vol_long_anual, color="red", linestyle="--")
                        ax.set_title(f"{melhor_nome} - {ativo}")
                        ax.legend()
                        st.pyplot(fig)

                    if vol_atual_anual > vol_long_anual * (1 + alarme_percent/100):
                        st.error(f"ALARME {ativo}: Vol Atual {vol_atual_anual:.2%} > Longo +{alarme_percent}%")

                    # CSV MT5 (gamma1 BRUTO sempre!)
                    csv_mt5 = io.StringIO()
                    csv_mt5.write("Parametro,Valor\n")
                    csv_mt5.write(f"omega,{omega}\n")
                    for i in range(1, melhor_p + 1):
                        a = res.params.get(f"alpha[{i}]", 0)
                        if a > 0: csv_mt5.write(f"alpha{i},{a}\n")
                    for i in range(1, melhor_q + 1):
                        b = res.params.get(f"beta[{i}]", 0)
                        if b > 0: csv_mt5.write(f"beta{i},{b}\n")
                    if gamma_param != 0:
                        csv_mt5.write(f"gamma1,{gamma_param}\n")  # BRUTO pro MQL5!
                    st.download_button(f"MT5 {ativo}", csv_mt5.getvalue(), f"{ativo}_PARAMETROS_MT5.csv", "text/csv")

                except Exception as e:
                    st.error(f"Erro {ativo}: {e}")

        relatorio += "="*160 + "\n\n"
        relatorio += "EXPLICAÇÃO DOS PARÂMETROS (Ω α β γ)\n"
        relatorio += "="*160 + "\n"
        relatorio += "Ω (Omega)   → Volatilidade de longo prazo (intercept)\n"
        relatorio += "            • Quanto menor, mais estável o ativo\n"
        relatorio += "            • Em EGARCH pode ser negativo (assimetria forte)\n\n"
        relatorio += "α (Alpha)   → Impacto total de choques recentes (soma de todos os α[i])\n"
        relatorio += "            • α alto → volatilidade reage forte a eventos\n"
        relatorio += "            • α baixo <0.07 → mercado maduro/forex\n"
        relatorio += "            • α + β ≈ 0.98 → vol de hoje explica 98% da vol amanhã\n\n"
        relatorio += "β (Beta)    → Persistência total da volatilidade (soma de todos os β[i])\n"
        relatorio += "            • β próximo de 1 → vol dura MUITO tempo\n"
        relatorio += "            • β > 0.98 → VOL DURA MUITO\n\n"
        relatorio += "γ (Gamma)   → Assimetria (efeito alavancagem)\n"
        relatorio += "            • Presente em: EGARCH e GJR-GARCH\n"
        relatorio += "            • γ > 0 → más notícias aumentam vol mais que boas\n"
        relatorio += "            • γ = 0 → sem assimetria (GARCH)\n"
        relatorio += "            • Se γ ≠ 0 → use EGARCH ou GJR no EA!\n\n"
        relatorio += "DICAS PARA MT5:\n"
        relatorio += "• EGARCH: use log(vol) → exp() no MQL5\n"
        relatorio += "• GJR: use (retorno < 0) ? (alpha + gamma) : alpha\n"
        relatorio += "• Para GARCH(p,q): some todos os α[i] e β[i]\n"
        relatorio += "• Atualize todo dia com novos dados\n"
        relatorio += "="*160 + "\n\n"
        relatorio += "LEGENDA DAS INTERPRETAÇÕES AUTOMÁTICAS (v3.9.4)\n"
        relatorio += "="*160 + "\n"
        relatorio += "FOREX CLÁSSICO     → FOREX + GARCH + α<0.07 + β>0.90\n"
        relatorio += "VOL TÉCNICA        → FUTUROS + GARCH + α>0.08\n"
        relatorio += "ACAO MADURA        → AÇÃO + GARCH + α<0.07\n"
        relatorio += "ACAO VOLÁTIL       → AÇÃO + GARCH + α>0.15\n"
        relatorio += "QUEDAS EXPLODEM VOL! → EGARCH + Ω < -0.5\n"
        relatorio += "VOL DURA MUITO     → β > 0.98\n"
        relatorio += "TECH/PÂNICO        → EGARCH + Ω < -0.3\n"

        st.code(relatorio, language="text")
        st.download_button("BAIXAR RELATÓRIO DIDÁTICO", relatorio, f"ANALISE_GARCH_PRO_{datetime.now().strftime('%Y-%m-%d')}.txt", "text/plain")
