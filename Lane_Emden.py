# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:27:24 2026

@author: KamCKPM
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
import seaborn as sns

### Configuración Inicial y Estilos Globales
filas_tabla = []
sns.set_theme(style="whitegrid")
sns.set_context("poster")

# Ruta de acceso al Modelo Solar Estándar (Model S de Christensen-Dalsgaard)
ruta = r"./Jorgen Christensen.txt"

### Lectura del archivo de datos: 
with open(ruta, "r") as f:
    lineas = f.readlines()
    
datos = [linea.split() for linea in lineas[1:]] 
nombres = ["r/R", "c (cm/sec)", "rho (g/cm^3)", "p (dyn/cm^2)", "Gamma_1", "T (K)"]
df = pd.DataFrame(datos, columns=nombres)
df = df.apply(pd.to_numeric, errors="coerce")

### Tabla para las primeras raices de los distintos m
def Calcular_Componentes_Tabla(m, x, solucion_bvp):
    """
    Calcula las constantes y componentes astrofísicos adimensionales de la 
    estrella politrópica a partir de la solución numérica continua de la EDO.
    """
    y_anterior = solucion_bvp.sol(x)[0]
    indices_debajo_cero = np.where(y_anterior <= 0)[0]
    
    if len(indices_debajo_cero) > 0 and m < 5.0:
        idx_cero_malla = indices_debajo_cero[0]
        
        # Función explícita para evaluar la componente y(r) en el buscador de raíces
        def evaluar_y_continua(r):
            componentes_solucion = solucion_bvp.sol(r)
            return componentes_solucion[0] 
        
        # Búsqueda del primer cero exacto (superficie estelar xi_1)
        limite_superior = x[idx_cero_malla]
        resultado_raiz = root_scalar(evaluar_y_continua, bracket=[1e-6, limite_superior])
        xi_1 = resultado_raiz.root
        
        # Determinación de la derivada exacta en la superficie
        derivada_exacta_funcion = solucion_bvp.sol.derivative(1)
        derivada_cero = derivada_exacta_funcion(xi_1)[0]
        
        # Cálculo de parámetros estructurales astrofísicos
        masa_adimensional = -(xi_1**2) * derivada_cero
        factor_densidad = xi_1 / (-3.0 * derivada_cero)
        
        if m != 1:
            omega_n = -(xi_1**((m+1)/(m-1))) * derivada_cero
        else:
            omega_n = np.nan
            
        if m != 0:
            N_n = (1/(m+1)) * (4*np.pi/(omega_n**(m-1)))**(1/m)
        else:
            N_n = np.nan
            
        W_n = 1.0 / (4.0 * np.pi * (m + 1.0) * (derivada_cero**2))
        
    else:
        # Soluciones asintóticas para índices sin superficie definida (m >= 5)
        xi_1 = np.inf
        masa_adimensional = np.nan
        factor_densidad = np.inf
        omega_n = np.nan
        N_n = np.nan
        W_n = np.nan

    return {
        'n': m,
        'xi_1': xi_1,
        '-xi_1^2 * theta_prime': masa_adimensional,
        'rho_c / rho_bar': factor_densidad,
        'omega_n': omega_n,
        'N_n': N_n,
        'W_n': W_n
    }


### Método numérico para la solución de Lane-Emden
def Lane_Emden(m):
    """
    Resuelve la ecuación  de Lane-Emden, distingue entre exponentes politropicos
    no enteros y evita indeterminaciones al realizar la gráfica
    """
    limite_y_negativo = -1.5
    x = np.linspace(1e-6, 17.5, 1000)
    y_anterior = np.ones_like(x)
    
    def EDO(x, Y):
        funcion_y = Y[0]
        derivada_y = Y[1]
        
        y_previa_interp = np.interp(x, malla_referencia, y_anterior)
        y_previa_valida = y_previa_interp

        a = m * (y_previa_valida ** (m - 1))
        b = (m - 1) * (y_previa_valida ** m)

        derivada_segunda_y = -(2 / x) * derivada_y - a * funcion_y + b
        return np.vstack((derivada_y, derivada_segunda_y))

    def Frontera(Inicial, Final):
        return np.array([Inicial[0]-1, Inicial[1]])
    
    malla_referencia = x.copy()

    # Bucle de linealización iterativa para convergencia de la EDO
    for iteracion in range(15):
        estado_inicial = np.vstack((y_anterior, np.zeros_like(x)))
        solucion_bvp = solve_bvp(EDO, Frontera, x, estado_inicial)
        y_anterior = solucion_bvp.sol(x)[0]
        malla_referencia = x.copy() 
        
    # Extracción y almacenamiento de componentes para la tabla comparativa
    componentes = Calcular_Componentes_Tabla(m, x, solucion_bvp)
    filas_tabla.append(componentes)
    
    # Criterio de recorte de la solución para evitar indeterminaciones en el plano gráfico
    es_no_entero = not float(m).is_integer()
    if es_no_entero:
        indices_recorte = np.where(y_anterior <= 0)[0]
    else:
        indices_recorte = np.where(y_anterior <= limite_y_negativo)[0]
        
    if len(indices_recorte) > 0:
        primer_cero_idx = indices_recorte[0]
        radio_grafico = x[:primer_cero_idx + 1]
        solucion_grafica = y_anterior[:primer_cero_idx + 1]
    else:
        radio_grafico = x
        solucion_grafica = y_anterior

    return x, solucion_bvp, radio_grafico, solucion_grafica

### Grafica la densidad respecto a la polítropa
def Graficar_Densidad_Solar(df_model_s, soluciones_calculadas, cambio=1, x_lim=1.02):
    """
    Grafica la densidad fraccionaria (rho / rho_c) del Sol (Model S)
    y lo compara con las soluciones de las polítropas dadas.
    """
    plt.figure(figsize=(10, 6))
    
    # Inversión de los datos del Model S (mapeo del centro r/R=0 hacia la superficie r/R=1)
    r_solar_fraccion = df_model_s["r/R"].values[::-1]
    densidad_solar = df_model_s["rho (g/cm^3)"].values[::-1]
    rho_c_solar = densidad_solar[0]
    
    plt.plot(r_solar_fraccion, densidad_solar / rho_c_solar, 
             label="Model S", color='black', lw=3.5)
    
    # Configuración de estilos de línea rotativos
    if cambio == 1:
        estilos_linea = ['--', '-.', ':', '-']
    else:
        estilos_linea = ['-']
        
    for i, (m, (x, solucion_bvp)) in enumerate(soluciones_calculadas.items()):
        y_malla = solucion_bvp.sol(x)[0]
        idx_cero_malla = np.where(y_malla <= 0)[0][0]
        limite_superior = x[idx_cero_malla]
        
        def evaluar_y_continua(r):
            return solucion_bvp.sol(r)[0]
            
        resultado_raiz = root_scalar(evaluar_y_continua, bracket=[1e-6, limite_superior])
        xi_1 = resultado_raiz.root
        
        x_perfil = np.linspace(1e-6, xi_1, 500)
        y_perfil = solucion_bvp.sol(x_perfil)[0]
        
        color_curva = plt.cm.tab20(i)
        estilo_actual = estilos_linea[i % len(estilos_linea)]
        
        # Relación de densidad para un polítropo: rho / rho_c = y^m
        plt.plot(x_perfil / xi_1, y_perfil ** m, 
                 label=f"m = {m}", color=color_curva, lw=3, linestyle=estilo_actual)
        
    plt.xlim(0, x_lim)
    plt.ylim(-0.02, 1.05)
    plt.xlabel(r"Fracción del Radio ($r / R_\odot$)")
    plt.ylabel(r"Densidad Fraccionaria ($\rho / \rho_c$)")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()
    
    # Oscurecimiento de los bordes del Canvas
    ax = plt.gca()
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
        
    plt.show()

### Grafica la presión respecto a la polítropa
def Graficar_Presion_Solar(df_model_s, soluciones_calculadas, cambio=1, x_lim=1.02):
    """
    Grafica la presión fraccionaria (P / P_c) del Sol (Model S)
    y lo compara con las soluciones de las polítropas dadas.
    """
    plt.figure(figsize=(10, 6))
    
    # Inversión de los datos del Model S (mapeo del centro r/R=0 hacia la superficie r/R=1)
    r_solar_fraccion = df_model_s["r/R"].values[::-1]
    presion_solar = df_model_s["p (dyn/cm^2)"].values[::-1]
    P_c_solar = presion_solar[0]
    
    plt.plot(r_solar_fraccion, presion_solar / P_c_solar, 
             label="Model S", color='black', lw=3.5)
    
    if cambio == 1:
        estilos_linea = ['--', '-.', ':', '-']
    else:
        estilos_linea = ['-']
    
    for i, (m, (x, solucion_bvp)) in enumerate(soluciones_calculadas.items()):
        y_malla = solucion_bvp.sol(x)[0]
        idx_cero_malla = np.where(y_malla <= 0)[0][0]
        limite_superior = x[idx_cero_malla]
        
        def evaluar_y_continua(r):
            return solucion_bvp.sol(r)[0]
            
        resultado_raiz = root_scalar(evaluar_y_continua, bracket=[1e-6, limite_superior])
        xi_1 = resultado_raiz.root
        
        x_perfil = np.linspace(1e-6, xi_1, 500)
        y_perfil = solucion_bvp.sol(x_perfil)[0]
        
        color_curva = plt.cm.tab20b(i)
        estilo_actual = estilos_linea[i % len(estilos_linea)]
        
        # Relación de presión para un polítropo: P / P_c = y^(m+1)
        plt.plot(x_perfil / xi_1, y_perfil ** (m + 1), 
                 label=f"m = {m}", color=color_curva, lw=3, linestyle=estilo_actual)
        
    plt.xlim(0, x_lim)
    plt.ylim(-0.02, 1.05)
    plt.xlabel("Fracción del Radio ($r / R_\odot$)")
    plt.ylabel(r"Presión Fraccionaria ($P / P_c$)")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()
    
    # Oscurecimiento de los bordes del Canvas
    ax = plt.gca()
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
        
    plt.show()
    
    
### Configuración de Conjuntos de Índices Politrópicos
Valores_m = [0.0, 0.26, 1.0, 1.79, 2.0, 2.37, 3.0, 3.14, 3.42, 3.81, 4.0] 
Valores_s = [3.17, 3.32, 3.47, 3.62, 3.77]

soluciones_grafica = {} 
soluciones_grafica_s = {}

indices_a_graficar = Valores_m
indices_a_graficar_s = Valores_s

### Compara los índices m seleccionados entre 0 y 4
def comparativa_politropas():
    """Ejecuta el análisis y gráficos para el espectro general de índices polítropos (m de 0 a 4)."""
    plt.figure(figsize=(10, 6))
    for m in Valores_m:
        x_calc, sol_bvp_calc, radio_grafico, solucion_grafica = Lane_Emden(m)
        plt.plot(radio_grafico, solucion_grafica, label=f"m = {m}", lw=3)
    
        if m in indices_a_graficar:
            soluciones_grafica[m] = (x_calc, sol_bvp_calc)

    plt.ylim(-0.6, 1.1) 
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta_n$")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    Graficar_Densidad_Solar(df, soluciones_grafica)
    Graficar_Presion_Solar(df, soluciones_grafica)
    
### Compara los índices m seleccionados entre 3.1 y 3.8
def comparativa_solar():
    """Ejecuta el análisis acotado enfocado en los polítropos de mejor ajuste al Sol (m ~ 3.1 a 3.8)."""
    plt.figure(figsize=(10, 6))
    for s in Valores_s:
        x_calc, sol_bvp_calc, radio_grafico, solucion_grafica = Lane_Emden(s)
        plt.plot(radio_grafico, solucion_grafica, label=f"m = {s}", lw=3)
    
        if s in indices_a_graficar_s:
            soluciones_grafica_s[s] = (x_calc, sol_bvp_calc)

    plt.ylim(-0.05, 1.1) 
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta_n$")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    Graficar_Densidad_Solar(df, soluciones_grafica_s, cambio=0, x_lim=0.5)
    Graficar_Presion_Solar(df, soluciones_grafica_s, cambio=0, x_lim=0.5)


def Tabla():
    """Resuelve las EDOs pendientes y genera la tabla resumen."""
    for m in Valores_m:
        Lane_Emden(m)
    df_tabla = pd.DataFrame(filas_tabla)
    formato_columnas = {'xi_1': '{:.3f}', '-xi_1^2 * theta_prime': '{:.3f}', 'rho_c / rho_bar': '{:.3f}', 'omega_n': '{:.3f}', 'W_n': '{:.3f}'}
    print(df_tabla.style.format(formato_columnas).hide(axis='index').to_string())


### Bloque de Activación de Funciones
# Descomentar la función requerida según las gráficas/tabla que se quiera visualizar:

# comparativa_politropas()
# comparativa_solar()
# Tabla()