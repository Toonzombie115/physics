import numpy as np
from scipy.linalg import eigh
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuracion
as_size = 200
n_state = 10
cutoff = 30
lambda_c = 0.2

# Rango de k (eje X)
k_range = (-3, 3)
delta_k = 0.005
k_values = np.round(np.arange(k_range[0], k_range[1] + delta_k, delta_k), 4)

# Construccion de operadores
def build_operators(as_size):
    n = np.arange(as_size)
    
    O1 = np.diag((1 + 2*n)/4)
    for i in range(as_size - 2):
        val = np.sqrt((i+1)*(i+2))/4
        O1[i, i+2] = O1[i+2, i] = val
    
    O2 = np.diag((1 + 2*n + 2*n**2)/32)
    for i in range(as_size - 2):
        val = (2*np.sqrt(i+1)*(i+2)**(3/2) - np.sqrt((i+1)*(i+2)))/48
        O2[i, i+2] = O2[i+2, i] = val
    for i in range(as_size - 4):
        val = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))/96
        O2[i, i+4] = O2[i+4, i] = val
    
    return O1, O2

# Funcion de calculo
def calculate_point(k):
    try:
        O1, O2 = build_operators(as_size)
        n = np.arange(as_size)
        
        H = np.diag((8 + 8*k*(1 + 2*n) + lambda_c + 2*n**2*lambda_c + 2*n*(8 + lambda_c))/32)
        
        for i in range(as_size - 2):
            val = (12*k*np.sqrt((i+1)*(i+2)) + 2*np.sqrt(i+1)*(i+2)**(3/2)*lambda_c - 
                  np.sqrt((i+1)*(i+2))*(12 + lambda_c))/48
            H[i, i+2] = H[i+2, i] = val
            
        for i in range(as_size - 4):
            val = np.sqrt((i+1)*(i+2)*(i+3)*(i+4))*lambda_c/96
            H[i, i+4] = H[i+4, i] = val
        
        eigvals, eigvecs = eigh(H)
        psi_mu = eigvecs[:, n_state]
        E_mu = eigvals[n_state]
        
        g11, g12, g22 = 0.0, 0.0, 0.0
        for m in range(min(cutoff, as_size)):
            if m == n_state:
                continue
            delta_E = E_mu - eigvals[m]
            if abs(delta_E) < 1e-10:
                continue
            inv_delta_E2 = 1.0 / (delta_E ** 2)
            
            psi_m = eigvecs[:, m]
            O1_mu_m = np.dot(psi_mu, O1 @ psi_m)
            O2_mu_m = np.dot(psi_mu, O2 @ psi_m)
            
            g11 += O1_mu_m**2 * inv_delta_E2
            g12 += O1_mu_m * O2_mu_m * inv_delta_E2
            g22 += O2_mu_m**2 * inv_delta_E2
        
        return (k, g11, g12, g22, E_mu)
    except Exception as e:
        print(f"Error en k={k}: {str(e)}")
        return (k, np.nan, np.nan, np.nan, np.nan)

if __name__ == '__main__':
    print(f"\nCalculando QMT para λ = {lambda_c} y estado n_state = {n_state}...")
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(calculate_point, k_values)
    
    # Procesar resultados
    df = pd.DataFrame(results, columns=['k', 'g11', 'g12', 'g22', 'energy'])
    df.dropna(inplace=True)
    
    # Guardar datos
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"QMT_data_lambda_{lambda_c}_state_{n_state}_as{as_size}_deltak{delta_k}.csv"
    df.to_csv(csv_filename, index=False)
    

    def create_plotly_plot(df, y_col, title, ylabel, color):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['k'],
            y=df[y_col],
            mode='lines',
            line=dict(color=color, width=2),
            name=ylabel,
            hovertemplate=(
                "<b>k</b>: %{x:.4f}<br>"
                f"<b>{ylabel}</b>: %{{y:.4e}}<br>"
                f"λ = {lambda_c}, Estado = {n_state}"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='k',
            yaxis_title=ylabel,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            template='plotly_white',
            margin=dict(l=50, r=50, b=50, t=80),
            height=600
        )
        
        fig.add_annotation(
            xref='paper', yref='paper',
            x=1, y=1.05,
            text=f"λ = {lambda_c} | Estado {n_state}",
            showarrow=False,
            font=dict(size=10)
        )
        
        html_filename = f"{y_col}_vs_k_{lambda_c}_state_{n_state}_as{as_size}_deltak{delta_k}.html"
        fig.write_html(html_filename, include_plotlyjs='cdn')
        
        return html_filename

    # Crear graficos para cada componente
    plots = [
        ('g11', '$g_{11}$', 'red', 'Componente g₁₁ del QMT'),
        ('g12', '$g_{12}$', 'blue', 'Componente g₁₂ del QMT'),
        ('g22', '$g_{22}$', 'green', 'Componente g₂₂ del QMT'),
        ('energy', 'Energía', 'purple', 'Energía del estado')
    ]

    html_files = []
    for col, ylabel, color, title in plots:
        html_files.append(create_plotly_plot(df, col, title, ylabel, color))

    fig_combined = make_subplots(rows=4, cols=1, subplot_titles=[
        f"Componentes del QMT y Energía (λ = {lambda_c}, Estado {n_state})"
    ] + [""]*3)

    for i, (col, ylabel, color, _) in enumerate(plots, 1):
        fig_combined.add_trace(
            go.Scatter(
                x=df['k'],
                y=df[col],
                mode='lines',
                line=dict(color=color, width=2),
                name=ylabel,
                hovertemplate=f"<b>k</b>: %{{x:.4f}}<br><b>{ylabel}</b>: %{{y:.4e}}<extra></extra>"
            ),
            row=i, col=1
        )
        fig_combined.update_yaxes(title_text=ylabel, row=i, col=1)

    fig_combined.update_xaxes(title_text='k', row=4, col=1)
    fig_combined.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    combined_filename = f"QMT_combined__{lambda_c}_state_{n_state}_as{as_size}_deltak{delta_k}.html"
    fig_combined.write_html(combined_filename, include_plotlyjs='cdn')
    print(f"\nDashboard combinado guardado en {combined_filename}")
    print(f"\nTiempo total de ejecución: {time.time() - start_time:.2f} segundos")