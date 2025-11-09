"""
ASUS GARCH PRO 2025 - VERSÃO COMPLETA
Login + Cálculo Simples + Analyzer Pro Multi-Ativos
+ PERSISTÊNCIA DE USUÁRIOS (salva em arquivo JSON)
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import hashlib
import time
import json
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="ASUS GARCH PRO", page_icon="📊", layout="wide")

# ==================== PERSISTÊNCIA DE USUÁRIOS ====================
USERS_FILE = "users_database.json"

def carregar_usuarios():
    """Carrega usuários do arquivo JSON"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def salvar_usuarios(users_dict):
    """Salva usuários no arquivo JSON"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users_dict, f, indent=2, ensure_ascii=False)

# Inicializa usuários (carrega do arquivo se existir)
if "users" not in st.session_state:
    st.session_state.users = carregar_usuarios()

if "logado" not in st.session_state:
    st.session_state.logado = None
# ==================== CONFIGURAÇÕES GLOBAIS ====================
MODELOS_ANALYZER = [
    ('GARCH', 1, 0, 1, 'GARCH(1,1)'),
    ('GARCH', 1, 0, 2, 'GARCH(1,2)'),
    ('GARCH', 2, 0, 1, 'GARCH(2,1)'),
    ('EGARCH', 1, 1, 1, 'EGARCH(1,1)'),
    ('EGARCH', 1, 1, 2, 'EGARCH(1,2)'),
    ('GJR', 1, 1, 1, 'GJR-GARCH(1,1,1)')
]

TICKER_MAP = {
    'MES=F': 'ES', 'MNQ=F': 'NQ', 'M2K=F': 'RTY', 'MYM=F': 'YM',
    'EURUSD=X': 'EURUSD', 'BRL=X': 'USDBRL'
}

# ==================== FUNÇÕES DO ANALYZER ====================
def baixar_dados(ticker, inicio, fim):
    for _ in range(5):
        try:
            df = yf.download(ticker, start=inicio, end=fim, progress=False, auto_adjust=True)
            close = df['Close'].dropna()
            if len(close) < 500:
                raise ValueError("Menos de 500 pontos")
            return close
        except:
            time.sleep(3)
    raise ValueError(f"Falha ao baixar {ticker}")

def calcular_retornos(precos):
    ret = np.log(precos / precos.shift(1)).dropna()
    return ret.replace([np.inf, -np.inf], np.nan).dropna()

def ljung_box_test(residuals_sq, lags=20):
    if len(residuals_sq) < lags * 2:
        lags = max(1, len(residuals_sq) // 4)
    try:
        lb = acorr_ljungbox(residuals_sq, lags=[lags], return_df=True)
        return lb.iloc[0]['lb_pvalue']
    except:
        return 0.0

def ajustar_modelo(retornos, vol_type, p, o, q):
    try:
        model = arch_model(retornos, vol=vol_type, p=p, o=o, q=q, dist='normal')
        res = model.fit(disp="off", options={'maxiter': 1000})
        Z2 = (res.resid / res.conditional_volatility).dropna() ** 2
        lb_p = ljung_box_test(Z2)
        return {
            'params': res.params,
            'aic': res.aic,
            'lb_p': lb_p,
            'success': True,
            'model_name': f"{vol_type}({p},{o},{q})" if o else f"{vol_type}({p},{q})"
        }
    except:
        return {'params': None, 'aic': np.inf, 'lb_p': 0.0, 'success': False, 'model_name': vol_type}

def selecionar_melhor_modelo(retornos, ticker):
    resultados = []
    for vol, p, o, q, nome in MODELOS_ANALYZER:
        res = ajustar_modelo(retornos, vol, p, o, q)
        res['nome_exibicao'] = nome
        resultados.append(res)
    
    validos = [r for r in resultados if r['success'] and r['lb_p'] > 0.05]
    if not validos:
        validos = [r for r in resultados if r['success']]
    if not validos:
        return {'model_name': 'FALHA', 'aic': 999}, []
    
    melhor = min(validos, key=lambda x: x['aic'])
    return melhor, resultados

def extrair_parametros(params):
    if params is None or len(params) == 0:
        return {'omega': 0, 'alpha_total': 0, 'beta_total': 0, 'gamma': 0}
    
    omega = params.get('omega', 0)
    alpha_total = sum(params.get(f'alpha[{i}]', 0) for i in range(1, 10) if f'alpha[{i}]' in params)
    beta_total = sum(params.get(f'beta[{i}]', 0) for i in range(1, 10) if f'beta[{i}]' in params)
    gamma = params.get('gamma[1]', 0)
    
    return {
        'omega': omega,
        'alpha_total': alpha_total,
        'beta_total': beta_total,
        'gamma': gamma
    }

def gerar_relatorio_txt_completo(resultados, inicio, fim, dias_corridos, dias_uteis):
    """Gera relatório TXT COMPLETO igual ao Jupyter"""
    width = 220
    lines = []
    
    lines.append("GARCH ANALYZER PRO 3.9.4 — ANÁLISE COMPLETA + REGRAS POR TIPO DE ATIVO")
    lines.append(f"Data da análise: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Período analisado: {inicio} → {fim}")
    lines.append(f"Dias corridos: {dias_corridos} | Dias úteis: {dias_uteis} (≈ {dias_uteis/252:.2f} anos)\n")
    
    lines.append("RESULTADOS DOS MODELOS VENCEDORES + INTERPRETAÇÃO AUTOMÁTICA")
    lines.append("=" * width)
    lines.append(f"{'Ativo':<8} {'Modelo':<16} {'AIC':<8} {'LB':<6} {'Ω':<12} {'α':<10} {'β':<10} {'γ':<10} {'Status':<10} {'Interpretação':<50}\n")
    lines.append("=" * width)
    
    for r in resultados:
        p = r['params']
        status = "EXCELENTE" if r['lb_p'] > 0.05 else "BOM"
        ativo, ticker = r['ativo'], r['ticker']
        
        omega = p.get('omega', p.get('mu', 0))
        alpha_total = sum(p.get(f'alpha[{i}]', 0) for i in range(1, 10) if f'alpha[{i}]' in p)
        beta_total = sum(p.get(f'beta[{i}]', 0) for i in range(1, 10) if f'beta[{i}]' in p)
        gamma = p.get('gamma[1]', 0.0)
        
        # Determina tipo de ativo
        tipo = "ACAO"
        if ativo in ['EURUSD', 'USDBRL'] or 'USD' in ticker or '=X' in ticker:
            tipo = "FOREX"
        elif '=F' in ticker or ativo in ['ES', 'NQ', 'RTY', 'YM']:
            tipo = "FUTUROS"
        elif ativo.startswith('^') or ativo in ['SPX', 'NDX', 'RUT']:
            tipo = "INDICE"
        
        # Regras de interpretação
        regras = []
        if r['model_name'].startswith('EGARCH'):
            if omega < -0.5: regras.append("QUEDAS EXPLODEM VOL!")
            elif omega < -0.2: regras.append("Quedas aumentam vol")
            elif omega < 0: regras.append("Leve alavancagem")
        if beta_total > 0.98: regras.append("VOL DURA MUITO (CRISES)")
        elif beta_total > 0.95: regras.append("Vol persistente")
        if alpha_total > 0.20: regras.append("REAÇÃO FORTE A NOTÍCIAS")
        elif alpha_total > 0.10: regras.append("Choques moderados")
        
        if tipo == "FOREX" and r['model_name'].startswith('GARCH') and alpha_total < 0.07 and beta_total > 0.90:
            regras.append("FOREX CLÁSSICO")
        elif tipo == "FUTUROS" and r['model_name'].startswith('GARCH') and alpha_total > 0.08:
            regras.append("VOL TÉCNICA (FUTUROS)")
        elif tipo == "ACAO" and r['model_name'].startswith('GARCH'):
            if alpha_total < 0.07: regras.append("ACAO MADURA")
            elif alpha_total > 0.15: regras.append("ACAO VOLÁTIL")
        
        if r['model_name'].startswith('EGARCH') and omega < -0.3:
            regras.append("TECH/PÂNICO")
        
        interp_str = " | ".join(regras) if regras else "Estável"
        
        lines.append(f"{ativo:<8} {r['model_name']:<16} {r['aic']:<8.1f} {r['lb_p']:<6.3f} "
                    f"{omega:<12.6f} {alpha_total:<10.6f} {beta_total:<10.6f} {gamma:<10.6f} {status:<10} {interp_str:<50}")
    
    lines.append("=" * width + "\n")
    
    # CRITÉRIOS DE SELEÇÃO
    lines.append("CRITÉRIOS DE SELEÇÃO DO MELHOR MODELO")
    lines.append("=" * width)
    lines.append("AIC (Akaike Information Criterion)")
    lines.append("    • Quanto MENOR, MELHOR o modelo")
    lines.append("    • Penaliza complexidade → evita overfitting")
    lines.append("    • Ex: AIC = -5109 → EXCELENTE")
    lines.append("    • Ex: AIC = -4000 → modelo pior\n")
    lines.append("LB p-val (Ljung-Box p-value)")
    lines.append("    • Testa se resíduos são 'ruído branco'")
    lines.append("    • p-val > 0.05 → MODELO VÁLIDO")
    lines.append("    • p-val < 0.05 → resíduos com padrão → MODELO RUIM")
    lines.append("    • Status 'EXCELENTE' = p-val > 0.05\n")
    
    # PARÂMETROS GREGOS
    lines.append("INTERPRETAÇÃO DOS PARÂMETROS GREGOS")
    lines.append("=" * width)
    lines.append("Ω (Omega)   → Volatilidade de longo prazo")
    lines.append("            • GARCH/GJR: sempre positivo")
    lines.append("            • EGARCH: pode ser NEGATIVO → quedas aumentam vol mais que subidas")
    lines.append("            • Ex: Ω = -0.645 → quedas geram PÂNICO de vol\n")
    lines.append("α (Alpha)   → Impacto total de choques recentes (soma de todos os α[i])")
    lines.append("            • α alto → volatilidade reage forte a eventos")
    lines.append("            • Ex: α = 0.341 → 34.1% do choque entra na vol\n")
    lines.append("β (Beta)    → Persistência total da volatilidade (soma de todos os β[i])")
    lines.append("            • β próximo de 1 → vol dura MUITO tempo")
    lines.append("            • Ex: β = 0.991 → vol dura ~30 dias")
    lines.append("            • α + β ≈ 0.98 → vol de hoje explica 98% da vol amanhã\n")
    lines.append("γ (Gamma)   → Assimetria (efeito alavancagem)")
    lines.append("            • Presente em: EGARCH e GJR-GARCH")
    lines.append("            • γ > 0 → más notícias aumentam vol mais que boas")
    lines.append("            • γ = 0 → sem assimetria (GARCH)")
    lines.append("            • Se γ ≠ 0 → use EGARCH ou GJR no EA!\n")
    lines.append("DICAS PARA MT5:")
    lines.append("• EGARCH: use log(vol) → exp() no MQL5")
    lines.append("• GJR: use (retorno < 0) ? (alpha + gamma) : alpha")
    lines.append("• Para GARCH(p,q): some todos os α[i] e β[i]")
    lines.append("• Atualize todo dia com novos dados")
    lines.append("=" * width + "\n")
    
    # LEGENDA
    lines.append("LEGENDA DAS INTERPRETAÇÕES AUTOMÁTICAS (v3.9.4)")
    lines.append("=" * width)
    lines.append("FOREX CLÁSSICO     → FOREX + GARCH + α<0.07 + β>0.90")
    lines.append("VOL TÉCNICA        → FUTUROS + GARCH + α>0.08")
    lines.append("ACAO MADURA        → AÇÃO + GARCH + α<0.07")
    lines.append("ACAO VOLÁTIL       → AÇÃO + GARCH + α>0.15")
    lines.append("QUEDAS EXPLODEM VOL! → EGARCH + Ω < -0.5")
    lines.append("VOL DURA MUITO     → β > 0.98")
    lines.append("TECH/PÂNICO        → EGARCH + Ω < -0.3")
    lines.append("=" * width)
    
    return "\n".join(lines)

def gerar_csv_mt5(resultados):
    dados = []
    for r in resultados:
        params = extrair_parametros(r['params'])
        dados.append({
            'Ativo': r['ativo'],
            'Modelo': r['model_name'],
            'Omega': params['omega'],
            'Alpha_Total': params['alpha_total'],
            'Beta_Total': params['beta_total'],
            'Gamma': params['gamma'],
            'AIC': r['aic'],
            'LB_pval': r['lb_p']
        })
    return pd.DataFrame(dados)

# ==================== SISTEMA DE LOGIN ====================
if "users" not in st.session_state:
    st.session_state.users = {}
if "logado" not in st.session_state:
    st.session_state.logado = None

if not st.session_state.logado:
    st.title("🔐 ASUS GARCH PRO 2025")
    st.markdown("**Sistema com Login + GARCH Analyzer Profissional**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 CADASTRO")
        with st.form("cadastro"):
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password")
            convite = st.text_input("Código de Convite", help="king2025 / petr4god / asuspro")
            if st.form_submit_button("Cadastrar"):
                if convite not in ["king2025", "petr4god", "asuspro"]:
                    st.error("❌ Convite inválido!")
                elif email in st.session_state.users:
                    st.error("❌ E-mail já usado!")
                else:
                    st.session_state.users[email] = {
                        "senha": hashlib.sha256(senha.encode()).hexdigest(),
                        "aprovado": False
                    }
                    salvar_usuarios(st.session_state.users)  # ← SALVA NO ARQUIVO
                    st.success("✅ Cadastro enviado! Aguarde aprovação.")
    
    with col2:
        st.header("🔑 LOGIN")
        with st.form("login"):
            email_l = st.text_input("E-mail")
            senha_l = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                if email_l in st.session_state.users and st.session_state.users[email_l]["aprovado"]:
                    if st.session_state.users[email_l]["senha"] == hashlib.sha256(senha_l.encode()).hexdigest():
                        st.session_state.logado = email_l
                        st.rerun()
                    else:
                        st.error("❌ Senha errada!")
                else:
                    st.error("❌ Usuário não aprovado ou inexistente.")
    
    # PAINEL ADMIN
    st.divider()
    with st.expander("👑 Painel Admin"):
        if st.text_input("Senha Admin", type="password", key="admin_pwd") == "asus2025":
            st.success("✅ ADMIN LOGADO!")
            st.subheader("📋 Usuários Pendentes:")
            pendentes = {e: d for e, d in st.session_state.users.items() if not d["aprovado"]}
            
            if not pendentes:
                st.info("Nenhum usuário pendente.")
            else:
                for email, data in pendentes.items():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"📧 {email}")
                    with c2:
                        if st.button("✅ APROVAR", key=email):
                            st.session_state.users[email]["aprovado"] = True
                            salvar_usuarios(st.session_state.users)  # ← SALVA NO ARQUIVO
                            st.success(f"Aprovado: {email}")
                            st.rerun()

else:
    # ==================== ÁREA LOGADA ====================
    st.sidebar.success(f"✅ Logado: {st.session_state.logado}")
    if st.sidebar.button("🚪 Sair"):
        st.session_state.logado = None
        st.rerun()
    
    st.sidebar.divider()
    
    # MENU DE NAVEGAÇÃO
    pagina = st.sidebar.radio(
        "📊 Escolha o Módulo:",
        ["🎯 Cálculo Simples", "🔬 Analyzer Pro (Multi-Ativos)"]
    )
    
    # ==================== PÁGINA 1: CÁLCULO SIMPLES ====================
    if pagina == "🎯 Cálculo Simples":
        st.title("🎯 ASUS GARCH - Cálculo Simples")
        st.markdown("**Análise rápida de volatilidade para 1 ativo**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ativo = st.text_input("Ativo", "PETR4.SA").upper()
        with col2:
            modelo = st.selectbox("Modelo", ["GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH", "T-Student"])
        with col3:
            periodo = st.slider("Dias", 100, 2000, 500)

        if st.button("🚀 CALCULAR VOLATILIDADE", type="primary"):
            with st.spinner("📊 Baixando dados..."):
                try:
                    data = yf.download(ativo, period=f"{periodo}d", progress=False)
                    df = data["Close"].pct_change().dropna() * 100
                    scaled = df * 100

                    if "T-Student" in modelo:
                        vol = scaled.rolling(100).std() * np.sqrt(252)
                        vol_long = vol.mean() / 100
                        vol_atual = vol.iloc[-1] / 100
                        res = None
                    else:
                        vol_type = "Garch" if "GARCH" in modelo else "EGarch" if "EGARCH" in modelo else "GJR"
                        am = arch_model(
                            scaled, 
                            dist="normal", 
                            vol=vol_type, 
                            p=1, 
                            o=1 if "GJR" in modelo else 0, 
                            q=1
                        )
                        res = am.fit(disp="off")
                        unconditional = res.params["omega"] / (
                            1 - res.params["alpha[1]"] - res.params["beta[1]"] - res.params.get("gamma[1]", 0) / 2
                        )
                        vol_long = np.sqrt(unconditional) / 100
                        vol_atual = np.sqrt(res.conditional_volatility.iloc[-1]) / 100

                    # MÉTRICAS
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📈 Vol Longo Prazo", f"{vol_long:.4%}")
                    with col2:
                        st.metric("📊 Vol Atual", f"{vol_atual:.4%}")
                    with col3:
                        delta = (vol_atual / vol_long - 1) * 100
                        st.metric("📉 Diferença", f"{delta:+.2f}%")

                    # GRÁFICO
                    fig, ax = plt.subplots(figsize=(12, 6))
                    if res is not None:
                        (np.sqrt(res.conditional_volatility.iloc[-200:]) / 100).plot(ax=ax, label="Vol Condicional")
                    else:
                        (vol.iloc[-200:] / 100).plot(ax=ax, label="Vol Rolling")
                    ax.axhline(vol_long, color="red", linestyle="--", label="Vol Longo Prazo")
                    ax.set_title(f"{modelo} - {ativo}", fontsize=14, fontweight='bold')
                    ax.set_ylabel("Volatilidade")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                    # DOWNLOAD CSV
                    csv = data[["Close"]].pct_change().to_csv()
                    st.download_button("💾 Baixar Retornos CSV", csv, f"{ativo}_retornos.csv")
                    
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
    
    # ==================== PÁGINA 2: ANALYZER PRO ====================
    else:
        st.title("🔬 GARCH ANALYZER PRO - Multi-Ativos")
        st.markdown("**Análise comparativa de múltiplos ativos com seleção automática do melhor modelo**")
        
        # SIDEBAR - Configurações
        with st.sidebar:
            st.subheader("⚙️ Configurações")
            
            # Ativos pré-definidos
            ativos_disponiveis = [
                'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA',
                'MES=F', 'MNQ=F', 'M2K=F', 'MYM=F',
                'EURUSD=X', 'BRL=X',
                'NVDA', 'TSLA', 'AAPL', 'MSFT'
            ]
            
            ativos_selecionados = st.multiselect(
                "📈 Selecione os Ativos:",
                options=ativos_disponiveis,
                default=['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
            )
            
            # Adicionar ativo customizado
            with st.expander("➕ Adicionar Ativo Customizado"):
                novo_ativo = st.text_input("Ticker:", placeholder="Ex: AAPL")
                if st.button("Adicionar") and novo_ativo:
                    if novo_ativo.upper() not in ativos_selecionados:
                        ativos_selecionados.append(novo_ativo.upper())
                        st.success(f"✅ {novo_ativo.upper()} adicionado!")
            
            st.divider()
            
            # Período
            st.subheader("📅 Período de Análise")
            hoje = datetime.today()
            col1, col2 = st.columns(2)
            with col1:
                inicio = st.date_input(
                    "Início:",
                    value=(hoje - timedelta(days=365*5)).date(),
                    max_value=hoje.date()
                )
            with col2:
                fim = st.date_input(
                    "Fim:",
                    value=hoje.date(),
                    max_value=hoje.date()
                )
        
        # Área principal
        if not ativos_selecionados:
            st.warning("⚠️ Selecione pelo menos um ativo na barra lateral")
        else:
            st.info(f"📊 {len(ativos_selecionados)} ativos selecionados")
            
            if st.button("🚀 EXECUTAR ANÁLISE COMPLETA", type="primary", use_container_width=True):
                
                inicio_str = inicio.strftime('%Y-%m-%d')
                fim_str = fim.strftime('%Y-%m-%d')
                dias_corridos = (fim - inicio).days
                dias_uteis = np.busday_count(inicio_str, fim_str)
                
                st.success(f"⏳ Analisando {len(ativos_selecionados)} ativos de {inicio_str} a {fim_str} ({dias_uteis} dias úteis)")
                
                resultados_finais = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Processar cada ativo
                for idx, ticker in enumerate(ativos_selecionados):
                    status_text.text(f"🔄 Processando {ticker}... ({idx + 1}/{len(ativos_selecionados)})")
                    progress_bar.progress((idx + 1) / len(ativos_selecionados))
                    
                    try:
                        precos = baixar_dados(ticker, inicio_str, fim_str)
                        retornos = calcular_retornos(precos)
                        melhor, todos = selecionar_melhor_modelo(retornos, ticker)
                        
                        ativo_mt5 = TICKER_MAP.get(ticker, ticker.replace('=X', '').replace('=F', ''))
                        
                        resultados_finais.append({
                            'ativo': ativo_mt5,
                            'ticker': ticker,
                            'model_name': melhor['model_name'],
                            'aic': melhor['aic'],
                            'lb_p': melhor['lb_p'],
                            'params': melhor['params']
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao processar {ticker}: {e}")
                
                status_text.text("✅ Análise concluída!")
                progress_bar.empty()
                
                if not resultados_finais:
                    st.error("❌ Nenhum resultado válido")
                else:
                    # RESULTADOS
                    st.success(f"✅ Análise concluída com sucesso! {len(resultados_finais)} ativos processados.")
                    
                    st.subheader("📊 Resultados dos Modelos")
                    
                    df_resultados = []
                    for r in resultados_finais:
                        params = extrair_parametros(r['params'])
                        status = "✅ EXCELENTE" if r['lb_p'] > 0.05 else "⚠️ BOM"
                        
                        # Interpretação
                        regras = []
                        if r['model_name'].startswith('EGARCH'):
                            if params['omega'] < -0.5:
                                regras.append("QUEDAS EXPLODEM VOL")
                        if params['beta_total'] > 0.98:
                            regras.append("VOL PERSISTENTE")
                        if params['alpha_total'] > 0.20:
                            regras.append("REAÇÃO FORTE")
                        
                        interp = " | ".join(regras) if regras else "Estável"
                        
                        df_resultados.append({
                            'Ativo': r['ativo'],
                            'Modelo': r['model_name'],
                            'AIC': f"{r['aic']:.1f}",
                            'LB p-val': f"{r['lb_p']:.3f}",
                            'Ω (Omega)': f"{params['omega']:.6f}",
                            'α (Alpha)': f"{params['alpha_total']:.6f}",
                            'β (Beta)': f"{params['beta_total']:.6f}",
                            'γ (Gamma)': f"{params['gamma']:.6f}",
                            'Status': status,
                            'Interpretação': interp
                        })
                    
                    st.dataframe(pd.DataFrame(df_resultados), use_container_width=True)
                    
                    # DOWNLOADS
                    st.subheader("💾 Exportar Resultados")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # TXT COMPLETO (IGUAL AO JUPYTER)
                        txt_completo = gerar_relatorio_txt_completo(resultados_finais, inicio_str, fim_str, dias_corridos, dias_uteis)
                        st.download_button(
                            label="📄 Download Relatório TXT COMPLETO",
                            data=txt_completo,
                            file_name=f"ANALISE_GARCH_PRO_{datetime.now().strftime('%Y-%m-%d')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        # CSV MT5
                        df_csv = gerar_csv_mt5(resultados_finais)
                        st.download_button(
                            label="📊 Download CSV para MT5",
                            data=df_csv.to_csv(index=False, sep=';', encoding='utf-8-sig'),
                            file_name=f"PARAMETROS-MT5-{datetime.now().strftime('%Y-%m-%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # EXPLICAÇÃO
                    with st.expander("📖 Entenda os Parâmetros"):
                        st.markdown("""
                        ### 📐 Parâmetros do Modelo
                        
                        **Ω (Omega)** - Volatilidade de longo prazo
                        - GARCH/GJR: sempre positivo
                        - EGARCH: pode ser negativo → quedas aumentam vol mais que subidas
                        
                        **α (Alpha)** - Impacto de choques recentes
                        - α alto → volatilidade reage forte a eventos
                        - Ex: α = 0.341 → 34.1% do choque entra na vol
                        
                        **β (Beta)** - Persistência da volatilidade
                        - β próximo de 1 → vol dura muito tempo
                        - Ex: β = 0.991 → vol persiste por semanas
                        
                        **γ (Gamma)** - Assimetria (efeito alavancagem)
                        - γ > 0 → más notícias aumentam vol mais que boas
                        - Presente em: EGARCH e GJR-GARCH
                        
                        ### 📊 Critérios de Seleção
                        
                        **AIC** (Akaike Information Criterion)
                        - Quanto MENOR, MELHOR o modelo
                        - Penaliza complexidade para evitar overfitting
                        
                        **LB p-val** (Ljung-Box p-value)
                        - p-val > 0.05 → modelo válido ✅
                        - p-val < 0.05 → resíduos com padrão ❌
                        """)
    
    # FOOTER
    st.divider()
    st.markdown("**ASUS GARCH PRO 2025** | Feito pelo Grok com carinho pro Asus 💙")
