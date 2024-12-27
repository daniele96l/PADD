import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Parametri
investimento_mensile = 1000  # Importo di investimento mensile
percentuale_fissa = 0.20  # 20% va all'investimento regolare
percentuale_buffer = 0.80  # 80% va al buffer

# Regole di attivazione per drawdown
trigger_drawdown = [0.00, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50, -1.01]
percentuale_extra = [0.00, 0.30, 0.20, 0.10, 0.05, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.00]

# Scarica i dati dello S&P 500
ticker = "^GSPC"
data_inizio = "2020-01-01"
data_fine = "2024-06-16"
dati_sp500 = yf.download(ticker, start=data_inizio, end=data_fine, interval="1d")
dati_sp500 = dati_sp500["Close"]
dati_mensili = dati_sp500.resample('M').last()


def calcola_drawdown(prezzi):
    massimo_rolling = prezzi.expanding().max()
    drawdown = (prezzi - massimo_rolling) / massimo_rolling
    return drawdown


def calcola_percentuale_investimento(drawdown):
    # Crea coppie (trigger, percentuale) e ordina per livello di drawdown
    coppie_trigger = list(zip(trigger_drawdown, percentuale_extra))
    coppie_trigger.sort()

    # Trova il livello di drawdown corrispondente
    for trigger, percentuale in coppie_trigger:
        if drawdown <= trigger:
            #print(f"Attivato a drawdown del {trigger:.2%}. Investimento extra dal buffer: {percentuale:.0%}.")
            return percentuale

    return 0.0

def calcola_max_drawdown(drawdowns):
    return drawdowns.min() * 100



def calcola_cagr(valore_finale, totale_investito, anni):
    return (((valore_finale / totale_investito) ** (1 / anni)) - 1) * 100


def calcola_volatilita(rendimenti):
    return rendimenti.std() * np.sqrt(12) * 100


def calcola_sharpe(rendimenti, tasso_privo_di_rischio):
    rendimento_annuo = (1 + rendimenti.mean()) ** 12 - 1
    volatilita_annua = rendimenti.std() * np.sqrt(12)
    return (rendimento_annuo - tasso_privo_di_rischio) / volatilita_annua

# Calculate drawdown volatility
def calcola_volatilita_drawdown(drawdowns):
    return drawdowns.std() * 100

# Calculate downside deviation (for Sortino Ratio)
def calcola_deviazione_negativa(rendimenti, soglia):
    rendimenti_negativi = rendimenti[rendimenti < soglia]
    deviazione_negativa = np.sqrt(np.mean((rendimenti_negativi - soglia) ** 2))
    return deviazione_negativa * np.sqrt(12)  # Annualize

# Calculate Sortino Ratio
def calcola_sortino(rendimenti, tasso_privo_di_rischio):
    rendimento_annuo = (1 + rendimenti.mean()) ** 12 - 1
    deviazione_negativa = calcola_deviazione_negativa(rendimenti, 0)
    return (rendimento_annuo - tasso_privo_di_rischio) / deviazione_negativa


def simula_strategie(dati_mensili, investimento_mensile):
    rendimenti_mensili = dati_mensili.pct_change().fillna(0)
    drawdowns = calcola_drawdown(dati_mensili)

    # Inizializza i risultati
    date = rendimenti_mensili.index
    valori_pad = []
    valori_padd = []
    investito_pad = []
    investito_padd = []
    buffer = 0  # Buffer di cassa per la strategia PADD
    valori_buffer = []  # Traccia il valore del buffer nel tempo

    valore_corrente_pad = 0
    valore_corrente_padd = 0
    totale_investito_pad = 0
    totale_investito_padd = 0

    for i in range(len(rendimenti_mensili)):
        # Strategia PAD
        totale_investito_pad += investimento_mensile
        valore_corrente_pad = valore_corrente_pad * (1 + rendimenti_mensili.iloc[i]) + investimento_mensile
        valori_pad.append(valore_corrente_pad)
        investito_pad.append(totale_investito_pad)

        # Strategia PADD
        # Investimento fisso (20%)
        investimento_fisso = investimento_mensile * percentuale_fissa
        totale_investito_padd += investimento_fisso

        # Aggiungi al buffer (80%)
        buffer += investimento_mensile * percentuale_buffer

        # Controlla il drawdown e investi dal buffer se attivato
        drawdown_corrente = drawdowns.iloc[i]
        percentuale_extra = calcola_percentuale_investimento(drawdown_corrente)
        investimento_extra = buffer * percentuale_extra
        buffer -= investimento_extra  # Riduci il buffer dell'importo investito
        totale_investito_padd += investimento_extra

        # Aggiorna il valore del portafoglio PADD
        valore_corrente_padd = valore_corrente_padd * (1 + rendimenti_mensili.iloc[i]) + investimento_fisso + investimento_extra
        valori_padd.append(valore_corrente_padd)
        investito_padd.append(totale_investito_padd)
        valori_buffer.append(buffer)

    return pd.DataFrame({
        'Data': date,
        'Valore PAD': valori_pad,
        'Valore PADD': valori_padd,
        'Investito PAD': investito_pad,
        'Investito PADD': investito_padd,
        'Buffer': valori_buffer,
        'Rendimenti PAD': pd.Series(valori_pad).pct_change().fillna(0),
        'Rendimenti PADD': pd.Series(valori_padd).pct_change().fillna(0)
    })

# Esegui la simulazione
risultati = simula_strategie(dati_mensili, investimento_mensile)

# Calcola il periodo in anni
inizio_periodo = risultati['Data'].iloc[0]
fine_periodo = risultati['Data'].iloc[-1]
anni = (fine_periodo - inizio_periodo).days / 365.25

# Calcola valori finali e metriche
valore_finale_pad = risultati['Valore PAD'].iloc[-1]
valore_finale_padd = risultati['Valore PADD'].iloc[-1]
totale_investito_pad = risultati['Investito PAD'].iloc[-1]
totale_investito_padd = risultati['Investito PADD'].iloc[-1]
buffer_rimanente = risultati['Buffer'].iloc[-1]

# Calcola CAGR
cagr_pad = calcola_cagr(valore_finale_pad, totale_investito_pad, anni)
cagr_padd = calcola_cagr(valore_finale_padd, totale_investito_padd, anni)

# Calcola valore del buffer con interesse composto al 3% annuo
interesse_annuo = 0.03
valore_buffer_con_interesse = risultati['Buffer'] * ((1 + interesse_annuo / 12) ** len(risultati['Buffer']))
valore_totale_padd_con_interesse = risultati['Valore PADD'] + valore_buffer_con_interesse

# Calcola statistiche per PADD con interesse del buffer
valore_finale_padd_con_interesse = valore_totale_padd_con_interesse.iloc[-1]
rendimento_totale_padd_con_interesse = ((valore_finale_padd_con_interesse / totale_investito_padd) - 1) * 100
cagr_padd_con_interesse = calcola_cagr(valore_finale_padd_con_interesse, totale_investito_padd, anni)

# Calcola volatilità
volatilita_pad = calcola_volatilita(risultati['Rendimenti PAD'])
volatilita_padd = calcola_volatilita(risultati['Rendimenti PADD'])
rendimenti_padd_con_buffer = valore_totale_padd_con_interesse.pct_change().fillna(0)
volatilita_padd_con_buffer = calcola_volatilita(rendimenti_padd_con_buffer)

# Calcola Sharpe Ratio
rendimento_privo_di_rischio = 0.03  # 3% annuo
sharpe_pad = calcola_sharpe(risultati['Rendimenti PAD'], rendimento_privo_di_rischio)
sharpe_padd = calcola_sharpe(risultati['Rendimenti PADD'], rendimento_privo_di_rischio)
sharpe_padd_con_buffer = calcola_sharpe(rendimenti_padd_con_buffer, rendimento_privo_di_rischio)

# Calcola metriche aggiuntive
max_drawdown_pad =  calcola_max_drawdown(calcola_drawdown(dati_mensili))


sortino_pad = calcola_sortino(risultati['Rendimenti PAD'], rendimento_privo_di_rischio)
sortino_padd = calcola_sortino(risultati['Rendimenti PADD'], rendimento_privo_di_rischio)
sortino_padd_con_buffer = calcola_sortino(rendimenti_padd_con_buffer, rendimento_privo_di_rischio)

# Print summarized metrics
print("\nStrategia PAC:")
print(f"Volatilità: {volatilita_pad:.2f}%")
print(f"CAGR: {cagr_pad:.2f}%")
print(f"Max Drawdown: {max_drawdown_pad:.2f}%")


print("\nStrategia PADD con Buffer e Interesse:")
print(f"Volatilità: {volatilita_padd_con_buffer:.2f}%")
print(f"CAGR: {cagr_padd_con_interesse:.2f}%")
print(f"Max Drawdown: {calcola_max_drawdown(calcola_drawdown(risultati['Valore PADD'])):.2f}%")

# Grafici
plt.style.use('seaborn')
fig = plt.figure(figsize=(12, 7))

# 1. Valori dei portafogli
ax1 = plt.subplot(2, 2, 1)
ax1.plot(risultati['Data'], risultati['Valore PAD'], label='PAC', linewidth=2)
ax1.plot(risultati['Data'], risultati['Valore PADD'], label='PADD', linewidth=2)
ax1.set_title('Valore dei Portafogli nel Tempo', fontsize=14, pad=15)
ax1.set_xlabel('Data', fontsize=12)
ax1.set_ylabel('Valore Portafoglio ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.tick_params(axis='both', which='major', labelsize=10)

# 2. Importi investiti
ax2 = plt.subplot(2, 2, 2)
ax2.plot(risultati['Data'], risultati['Investito PAD'], label='Investito PAC', linewidth=2)
ax2.plot(risultati['Data'], risultati['Investito PADD'], label='Investito PADD', linewidth=2)
ax2.set_title('Importi Investiti nel Tempo', fontsize=14, pad=15)
ax2.set_xlabel('Data', fontsize=12)
ax2.set_ylabel('Importo Investito ($)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.tick_params(axis='both', which='major', labelsize=10)

# 3. Valore del buffer e drawdown
ax3 = plt.subplot(2, 2, 3)
ax3.plot(risultati['Data'], risultati['Buffer'], label='Buffer di Cassa', color='green', linewidth=2)
ax3.set_xlabel('Data', fontsize=12)
ax3.set_ylabel('Valore Buffer ($)', fontsize=12, color='green')
ax3.tick_params(axis='y', labelcolor='green')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

ax3_twin = ax3.twinx()
drawdown = calcola_drawdown(dati_mensili)
ax3_twin.plot(risultati['Data'], drawdown, label='Drawdown Mercato', color='red', linewidth=2, linestyle='--')
ax3_twin.set_ylabel('Drawdown %', fontsize=12, color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')
ax3_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

linee1, etichette1 = ax3.get_legend_handles_labels()
linee2, etichette2 = ax3_twin.get_legend_handles_labels()
ax3.legend(linee1 + linee2, etichette1 + etichette2, loc='upper right')

ax3.set_title('Valore del Buffer e Drawdown di Mercato', fontsize=14, pad=15)
ax3.grid(True)

# 4. Valore PADD + Buffer con interesse
ax4 = plt.subplot(2, 2, 4)
ax4.plot(risultati['Data'], risultati['Valore PADD'], label='Valore PADD', linewidth=2)
ax4.plot(risultati['Data'], valore_buffer_con_interesse, label='Buffer con Interesse', linewidth=2, color='green')
# Aggiungi linea per valore totale (PADD + Buffer con interesse)
ax4.plot(risultati['Data'], valore_totale_padd_con_interesse, label='Valore Totale (PADD + Buffer con Interesse)', linewidth=2, color='purple', linestyle='--')
ax4.plot(risultati['Data'], risultati['Valore PAD'], label='PAC', linewidth=2, linestyle='-.')
ax4.set_title('Valore PADD + Buffer con Confronto PAC', fontsize=14, pad=15)
ax4.set_xlabel('Data', fontsize=12)
ax4.set_ylabel('Valore ($)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax4.tick_params(axis='both', which='major', labelsize=10)

fig.suptitle('Analisi delle Strategie di Investimento', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()
