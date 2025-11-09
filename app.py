"""
GARCH ANALYZER PRO 3.9.5 - STREAMLIT VERSION
Conversão do Jupyter Notebook para Streamlit
- Interface web moderna
- Mesma funcionalidade do notebook
- Roda local ou deploy gratuito
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import time
import os

# ==================== CONFIGURAÇÕES ====================
st.set_page_config(
    page_title="GARCH Analyzer Pro",
    page_icon="📊",
    layout="wide"
)

MODELOS = [
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

ARQUIVO_ATIVOS = "meus_ativos.txt"

# ==================== FUNÇÕES BÁSICAS ====================
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
    except Exception as e:
        return {'params': None, 'aic': np.inf, 'lb_p': 0.0, 'success': False, 'model_name': vol_type}

def selecionar_melhor_modelo(retornos, ticker):
    resultados = []
    for vol, p, o, q, nome in MODELOS:
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

# ==================== GERENCIAMENTO DE ATIVOS ====================
def carregar_ativos():
    if os.path.exists(ARQUIVO_ATIVOS):
        with open(ARQUIVO_ATIVOS, 'r', encoding='utf-8') as f:
            return [line.strip().upper() for line in f if line.strip()]
    else:
        return ['MES=F', 'MNQ=F', 'M2K=F', 'MYM=F', 'EURUSD=X', 'NVDA', 'TSLA']

def salvar_ativos(ativos):
    with open(ARQUIVO_ATIVOS, 'w', encoding='utf-8') as f:
        for ativo in sorted(ativos):
            f.write(ativo + '\n')

# ==================== EXPORTAÇÃO ====================
def gerar_relatorio_txt(resultados, inicio, fim, dias_corridos, dias_uteis):
    """Gera relatório TXT em memória"""
    width = 220
    lines = []
    
    lines.append("GARCH ANALYZER PRO 3.9.5 - ANÁLISE COMPLETA")
    lines.append(f"Data da análise: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Período analisado: {inicio} → {fim}")
    lines.append(f"Dias corridos: {dias_corridos} | Dias úteis: {dias_uteis} (≈ {dias_uteis/252:.2f} anos)\n")
    
    lines.append("RESULTADOS DOS MODELOS")
    lines.append("=" * width)
    lines.append(f"{'Ativo':<8} {'Modelo':<16} {'AIC':<8} {'LB':<6} {'Ω':<12} {'α':<10} {'β':<10} {'γ':<10} {'Status':<10} {'Interpretação':<50}")
    lines.append("=" * width)
    
    for r in resultados:
        p = r['params']
        status = "EXCELENTE" if r['lb_p'] > 0.05 else "BOM"
        params = extrair_parametros(p)
        
        omega = params['omega']
        alpha_total = params['alpha_total']
        beta_total = params['beta_total']
        gamma = params['gamma']
        
        # Interpretação
        regras = []
        if r['model_name'].startswith('EGARCH'):
            if omega < -0.5: regras.append("QUEDAS EXPLODEM VOL!")
            elif omega < -0.2: regras.append("Quedas aumentam vol")
        if beta_total > 0.98: regras.append("VOL DURA MUITO")
        if alpha_total > 0.20: regras.append("REAÇÃO FORTE A NOTÍCIAS")
        
        interp_str = " | ".join(regras) if regras else "Estável"
        
        lines.append(f"{r['ativo']:<8} {r['model_name']:<16} {r['aic']:<8.1f} {r['lb_p']:<6.3f} "
                    f"{omega:<12.6f} {alpha_total:<10.6f} {beta_total:<10.6f} {gamma:<10.6f} {status:<10} {interp_str:<50}")
    
    lines.append("=" * width)
    
    return "\n".join(lines)

def gerar_csv_mt5(resultados):
    """Gera CSV para MT5"""
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

# ==================== INTERFACE STREAMLIT ====================
def main():
    st.title("📊 GARCH ANALYZER PRO 3.9.5")
    st.markdown("**Análise de Volatilidade com GARCH, EGARCH e GJR-GARCH**")
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Gerenciamento de ativos
        st.subheader("📈 Ativos")
        if 'ativos' not in st.session_state:
            st.session_state.ativos = carregar_ativos()
        
        # Adicionar novo ativo
        col1, col2 = st.columns([3, 1])
        with col1:
            novo_ativo = st.text_input("Adicionar ativo:", placeholder="Ex: AAPL", key="novo_ativo")
        with col2:
            if st.button("➕", help="Adicionar"):
                if novo_ativo.strip():
                    ativo_upper = novo_ativo.strip().upper()
                    if ativo_upper not in st.session_state.ativos:
                        st.session_state.ativos.append(ativo_upper)
                        st.session_state.ativos = sorted(st.session_state.ativos)
                        salvar_ativos(st.session_state.ativos)
                        st.success(f"✅ {ativo_upper} adicionado!")
                        st.rerun()
        
        # Selecionar ativos
        ativos_selecionados = st.multiselect(
            "Selecione os ativos para análise:",
            options=st.session_state.ativos,
            default=st.session_state.ativos[:3]
        )
        
        # Remover ativos
        if st.button("🗑️ Remover selecionados"):
            for ativo in ativos_selecionados:
                st.session_state.ativos.remove(ativo)
            salvar_ativos(st.session_state.ativos)
            st.success("✅ Ativos removidos!")
            st.rerun()
        
        st.divider()
        
        # Período de análise
        st.subheader("📅 Período")
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
        return
    
    # Botão de análise
    if st.button("🚀 EXECUTAR ANÁLISE", type="primary", use_container_width=True):
        
        inicio_str = inicio.strftime('%Y-%m-%d')
        fim_str = fim.strftime('%Y-%m-%d')
        dias_corridos = (fim - inicio).days
        dias_uteis = np.busday_count(inicio_str, fim_str)
        
        st.info(f"📊 Analisando {len(ativos_selecionados)} ativos de {inicio_str} a {fim_str} ({dias_uteis} dias úteis)")
        
        resultados_finais = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(ativos_selecionados):
            status_text.text(f"Processando {ticker}...")
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
            return
        
        # Exibir resultados
        st.success(f"✅ Análise concluída com sucesso! {len(resultados_finais)} ativos processados.")
        
        # Tabela de resultados
        st.subheader("📊 Resultados dos Modelos")
        
        df_resultados = []
        for r in resultados_finais:
            params = extrair_parametros(r['params'])
            status = "✅ EXCELENTE" if r['lb_p'] > 0.05 else "⚠️ BOM"
            
            df_resultados.append({
                'Ativo': r['ativo'],
                'Modelo': r['model_name'],
                'AIC': f"{r['aic']:.1f}",
                'LB p-val': f"{r['lb_p']:.3f}",
                'Ω (Omega)': f"{params['omega']:.6f}",
                'α (Alpha)': f"{params['alpha_total']:.6f}",
                'β (Beta)': f"{params['beta_total']:.6f}",
                'γ (Gamma)': f"{params['gamma']:.6f}",
                'Status': status
            })
        
        st.dataframe(pd.DataFrame(df_resultados), use_container_width=True)
        
        # Downloads
        st.subheader("💾 Exportar Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # TXT
            txt_content = gerar_relatorio_txt(resultados_finais, inicio_str, fim_str, dias_corridos, dias_uteis)
            st.download_button(
                label="📄 Download Relatório TXT",
                data=txt_content,
                file_name=f"ANALISE_GARCH_{datetime.now().strftime('%Y-%m-%d')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # CSV MT5
            df_csv = gerar_csv_mt5(resultados_finais)
            st.download_button(
                label="📊 Download CSV para MT5",
                data=df_csv.to_csv(index=False, sep=';', encoding='utf-8-sig'),
                file_name=f"PARAMETROS-MT5-{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv"
            )
        
        # Explicação dos parâmetros
        with st.expander("📖 Entenda os Parâmetros"):
            st.markdown("""
            **Ω (Omega)** - Volatilidade de longo prazo
            - GARCH/GJR: sempre positivo
            - EGARCH: pode ser negativo → quedas aumentam vol
            
            **α (Alpha)** - Impacto de choques recentes
            - α alto → volatilidade reage forte a eventos
            - Ex: α = 0.341 → 34.1% do choque entra na vol
            
            **β (Beta)** - Persistência da volatilidade
            - β próximo de 1 → vol dura muito tempo
            - Ex: β = 0.991 → vol dura ~30 dias
            
            **γ (Gamma)** - Assimetria (efeito alavancagem)
            - γ > 0 → más notícias aumentam vol mais que boas
            - Presente em: EGARCH e GJR-GARCH
            
            **AIC** - Akaike Information Criterion
            - Quanto MENOR, MELHOR o modelo
            
            **LB p-val** - Ljung-Box p-value
            - p-val > 0.05 → modelo válido ✅
            - p-val < 0.05 → modelo ruim ❌
            """)

if __name__ == "__main__":
    main()
