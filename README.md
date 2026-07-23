# Método GT — Sistema de Análisis Estadístico MLB

**Última actualización: 23 de julio, 2026**

⚠️ **Este es un proyecto de investigación y aprendizaje estadístico. No hay capital real en juego. No se realizan apuestas hasta que el modelo demuestre rigor estadístico suficiente (mínimo 15-20 jornadas de validación por categoría).**

---

## 1. Qué es esto

Motor de análisis de partidos de MLB que usa simulación Monte Carlo para estimar probabilidades de victoria, handicap, totales (Over/Under) y F5 (primeras 5 entradas), comparándolas contra las cuotas de mercado para detectar posibles divergencias.

## 2. Reglas fundamentales del proyecto

- **Nunca sobreajustar con pocos datos**: mínimo 15-20 jornadas por categoría antes de sacar conclusiones o modificar parámetros.
- **Nunca forzar un pick cuando no hay edge genuino** ("no buscar acción").
- **Registrar TODO** en la base de datos, no solo lo que se juega, para poder calcular Brier Score y calibración real después.
- **Toda mejora debe validarse con backtesting fuera de muestra** (train en una temporada, test en otra nunca vista) antes de integrarse a producción.
- **Una variable nueva a la vez**: no agregar features más rápido de lo que se pueden validar.

---

## 3. Arquitectura

| Componente | Detalle |
|---|---|
| Código | Este repositorio (GitHub) |
| Ejecución automática | GitHub Actions, workflow "Corrida Diaria Metodo GT", cron diario ~1pm hora RD (`0 17 * * *` UTC) |
| Base de datos | Supabase (Postgres) |
| Origen histórico | Migrado desde Google Colab (abandonado por pérdida de datos en reinicios de runtime) |

### Archivos principales
- `motor.py` — todas las funciones del modelo (cálculo de runs esperados, simulación, calibración, matchup L/R, win%, contexto cualitativo)
- `corrida_diaria.py` — script que orquesta el análisis diario y guarda en Supabase
- `backtesting_xfip.py`, `backtesting_pesos.py` — scripts de validación histórica (2022-2023)
- `rescatar_historico.py` — recuperación puntual de datos de jornadas específicas
- `analizar_pesos.py`, `isotonic_test.py` — scripts de análisis estadístico sobre datos ya recolectados

### Tablas en Supabase
- `odds_snapshots` — cuotas y probabilidades del modelo por partido/mercado, con `tipo_snapshot` ('apertura'/'cierre') para cálculo de CLV
- `diagnostico_total` — seguimiento del mercado Total (modelo vs. mercado vs. resultado real)
- `backtesting_resultados` — resultados del backtesting xFIP vs. ERA-shrink (3,642 juegos, 2022-2023)
- `backtesting_pesos` — componentes individuales (f_era, f_kbb) para el experimento de optimización de pesos

---

## 4. Cómo funciona el modelo (resumen)

Para cada partido:
1. ERA del abridor con shrinkage bayesiano (hacia promedio de liga, según innings lanzados)
2. Combinado con K-BB% del abridor (pesos fijos: 0.70 ERA / 0.30 K-BB%)
3. Combinado con ERA de bullpen reciente, winsorizado a 9.00
4. Combinado con ofensiva del rival (exit velocity + barrel rate de Baseball Savant)
5. Ajustado por park factor
6. Simulado 20,000 veces vía Binomial Negativa
7. Calibrado con Platt Scaling (A=0.5795, B=0.1449, entrenado sobre 4,391 juegos 2022-2023)
8. Comparado contra cuota de mercado (vig removido) → bandera "alineado" / "moderada" / "extrema"

---

## 5. Experimentos de optimización ya realizados — NO REABRIR sin evidencia nueva

Cuatro intentos consecutivos de mejorar el modelo base fallaron o no se sostuvieron bajo validación rigurosa. **No repetir estos experimentos sin datos nuevos:**

| # | Experimento | Resultado |
|---|---|---|
| 1 | xFIP vs. ERA-shrink (3,642 juegos, 2022-2023) | ERA-shrink gana: 54.9% vs 50.7% precisión |
| 2 | Isotonic Regression vs. Platt Scaling | Platt gana: Brier 0.24615 vs 0.24650 |
| 3 | Pesos optimizados por regresión logística (sin separación temporal) | Parecía ganar, pero era espejismo |
| 4 | Pesos optimizados con validación train/val/test correcta (train 2022 primera mitad, val segunda mitad, test 2023 intacto) | Pesos fijos ganan: 0.24885 vs 0.24936 |

**Conclusión:** el cuello de botella no está en la métrica del abridor, ni en la calibración, ni en los pesos de combinación actuales. El modelo probablemente ya captura casi toda la señal disponible en estas variables. La mejora futura debe venir de **información nueva**, no de reoptimizar lo existente.

---

## 6. Variables en observación (NO integradas a la probabilidad todavía)

| Variable | Fecha de inicio | Estado |
|---|---|---|
| Matchup L/R (OPS del abridor vs. zurdos/diestros × composición del lineup rival) | 22 julio 2026 | Observación, solo se imprime/guarda |
| Win% ratio (récord de equipo vía standings MLB) | 23 julio 2026 | Observación, solo se imprime/guarda |

**No integrar a la probabilidad hasta acumular 15-20 jornadas de datos y validar aporte real fuera de muestra.**

---

## 7. Diagnóstico abierto: sesgo en el mercado de Total

Detectado desde el 18 de julio: el modelo **subestima sistemáticamente partidos de alta anotación** (11+ carreras).

**Hipótesis de causa raíz (auditoría de código, no validada):**
1. Simulación trata las carreras de ambos equipos como independientes, sin correlación compartida (clima, extra innings, umpire)
2. `DISPERSION_RUNS = 1.2` nunca se validó específicamente para el mercado de Total (se usa igual que para moneyline)

**Estado:** diagnosticado, no corregido. Requiere completar 15-20 jornadas antes de ajustar parámetros. Progreso: **5 jornadas registradas al 23 de julio** (43 partidos).

---

## 8. Roadmap priorizado — próxima etapa

**Secuencia recomendada — una variable a la vez, con validación completa antes de la siguiente:**

1. Terminar de validar matchup L/R y win% (en curso)
2. **Lineup confirmado** (WAR/OPS del lineup real vs. promedio del equipo) — extensión barata de infraestructura ya existente
3. **Fatiga real de bullpen** (lanzamientos últimos 3 días, descanso del cerrador) — nota: evidencia académica sugiere efecto real pero modesto (~0.007 ERA por pitcheo del día anterior), calibrar expectativas
4. Matchups ofensivos avanzados, defensa (OAA/DRS), catcher framing, umpire, clima, viajes — prioridad menor, evaluar en orden
5. **Método GT 2.0** (simulación por aparición al plato en vez de solo runs) — línea de investigación de largo plazo, no reemplazo inmediato

**Principio rector:** toda variable nueva debe demostrar, con backtesting fuera de muestra, que aporta señal predictiva que el modelo actual no captura. Si no puede demostrarse, no se integra.

---

## 9. Historial de incidencias técnicas relevantes

- **17-20 julio**: reinicios de runtime en Google Colab causaron pérdida de cache de pitchers y datos de sesión — motivó la migración completa a GitHub Actions + Supabase.
- **22 julio**: workflow de matchup L/R falló por timeout indefinido en `obtener_clima()` (Open-Meteo) — corregido con `timeout=10`.
- **23 julio**: bug en `obtener_winpct_equipo()` — comparaba por `abbreviation` cuando el JSON de standings solo expone `id` — corregido para comparar por `team_id`.
- **23 julio**: duplicados en `backtesting_resultados` y `diagnostico_total` por ejecuciones manuales repetidas del mismo workflow — limpiados vía `DELETE` conservando el registro más reciente por `game_id`.
