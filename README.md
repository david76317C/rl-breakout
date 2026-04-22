# 🎮 RL Breakout
Agente de aprendizaje por refuerzo entrenado sobre el ambiente **ALE/Breakout-v5** de Gymnasium (Arcade Learning Environment). El proyecto implementa un algoritmo Deep Q-Network (DQN) con PyTorch desde cero, utilizando una CNN para procesar fotogramas crudos del juego, junto con una interfaz de línea de comandos completa para entrenar, evaluar y visualizar el agente.

---

## 🗺️ El Ambiente

Breakout es el clásico juego de Atari (1976). El jugador controla una paleta en la parte inferior de la pantalla, rebotando una pelota para romper bloques de colores en la parte superior. El jugador pierde una vida si la pelota cae al fondo, y el episodio termina al perder las 5 vidas.

El estado es una imagen RGB de 210×160 píxeles, el agente no recibe ninguna información semántica sobre dónde está la pelota, la paleta o los bloques. Debe aprender a extraer toda esa información directamente de los píxeles, lo que hace necesaria una red neuronal convolucional (CNN).

---

## 📐 Espacio de Observaciones y Acciones

### Observaciones

El espacio de observación es una imagen RGB de dimensiones `(210, 160, 3)` — cada fotograma del juego tal como aparece en pantalla. Para que la CNN pueda procesarlos eficientemente, cada fotograma se preprocesa, se convierte a escala de grises, se redimensiona a 84×84 píxeles y se normaliza a valores entre 0 y 1. Además, se apilan los últimos 4 fotogramas como canales, resultando en un tensor de entrada `(4, 84, 84)`. Este apilamiento le da al agente percepción de movimiento, un solo fotograma no contiene información sobre la velocidad ni la dirección de la pelota.

### Acciones

El espacio de acciones es 4:

| Acción | Efecto | Significado |
|---|---|
| 0 | NOOP | no hacer nada |
| 1 | FIRE |lanzar la pelota al inicio de cada vida |
| 2 | Derecha | mover la paleta a la derecha |
| 3 | Izquierda | mover la paleta a la izquierda |

La acción FIRE es particularmente importante: sin ella la pelota nunca se lanza y el agente se queda parado indefinidamente. El agente debe descubrir por sí mismo que necesita presionar FIRE al inicio de cada vida.

### Recompensas

| Situación | Recompensa |
|---|---|
| Romper un bloque | +1 |
| No romper nada | 0 |
| Perder una vida | 0 (pero se pierde una de las 5 vidas) |
| Perder todas las vidas | Episodio termina |

Un episodio perfecto puede dar hasta aproximadamente 432 puntos (todos los bloques destruidos).

---

## 🔄 Flujo de Entrenamiento

### DQN con CNN

```
1. Inicializar red neuronal Q (CNN) y red objetivo
2. Para cada episodio:
   a. Preprocesar observación → 4 fotogramas grises apilados (4, 84, 84)
   b. Seleccionar acción con política ε-greedy
   c. Ejecutar acción → obtener recompensa y nuevo fotograma
   d. Guardar (s, a, r, s', done) en ReplayBuffer
   e. Cada 4 pasos → samplear mini-batch del buffer y actualizar pesos
   f. Cada 1,000 pasos → sincronizar red objetivo
3. Reducir ε gradualmente con decay multiplicativo
```

**Particularidades clave:**

El estado es un tensor multidimensional (píxeles). No hay conversión intermedia, la CNN recibe directamente los fotogramas preprocesados y aprende por sí misma a extraer las features relevantes (posición de la pelota, posición de la paleta, disposición de los bloques) a través de sus filtros convolucionales.

El ReplayBuffer almacena las transiciones en formato uint8 (enteros de 0 a 255) en lugar de float32 para ahorrar memoria RAM. Cada transición contiene dos estados de `(4, 84, 84)`, y con 100,000 transiciones almacenadas.

---

## 🧠 Arquitectura de la Red Neuronal (DQN)

Se utiliza la siguiente arquitectura CNN:

```
Entrada (4, 84, 84)  → Conv2d(32, 8×8, stride=4) → ReLU
                     → Conv2d(64, 4×4, stride=2) → ReLU
                     → Conv2d(64, 3×3, stride=1) → ReLU
                     → Flatten(3136)
                     → Linear(512) → ReLU
                     → Linear(4)
```

| Capa | Entrada | Salida | Kernel | Stride | Activación |
|---|---|---|---|---|---|
| Conv1 | 4 canales | 32 filtros | 8×8 | 4 | ReLU |
| Conv2 | 32 filtros | 64 filtros | 4×4 | 2 | ReLU |
| Conv3 | 64 filtros | 64 filtros | 3×3 | 1 | ReLU |
| FC1 | 3,136 | 512 | — | — | ReLU |
| FC2 (salida) | 512 | 4 (Q-values) | — | — | Ninguna |

**¿Por qué esta arquitectura?** Breakout es un ambiente visual donde toda la información está en los píxeles. Las capas convolucionales preservan las relaciones espaciales entre píxeles vecinos, permitiendo que la red detecte bordes, formas y patrones de movimiento. La red tiene aproximadamente 1.7 millones de parámetros.

### Hiperparámetros

| Parámetro | Valor |
|---|---|
| Learning rate | 1e-4 |
| Gamma (γ) | 0.99 |
| Epsilon inicio | 1.0 |
| Epsilon fin | 0.05 |
| Epsilon decay | 0.9995 |
| Batch size | 32 |
| Buffer capacity | 100,000 |
| Target update | cada 1,000 pasos |
| Learn every | cada 4 pasos |
| Min buffer | 10,000 transiciones |
| Función de pérdida | SmoothL1Loss (Huber) |
| Optimizer | Adam |
| Gradient clipping | max_norm = 10.0 |

---

## 📊 Resultados del Entrenamiento

### DQN (7,500 episodios)

Al inicio del entrenamiento el agente obtenía un promedio de 1.5 bloques por partida con un mejor puntaje de 8, jugando prácticamente al azar con epsilon en alrededor de 0.90. 

```
Episode  50/7500 | Avg Reward: 1.60 | Best Reward: 6.00 | Epsilon: 0.9753 | Buffer: 10,078
Episode 100/7500 | Avg Reward: 1.12 | Best Reward: 6.00 | Epsilon: 0.9512 | Buffer: 19,230
Episode 150/7500 | Avg Reward: 1.54 | Best Reward: 7.00 | Epsilon: 0.9277 | Buffer: 28,995
Episode 200/7500 | Avg Reward: 1.60 | Best Reward: 7.00 | Epsilon: 0.9048 | Buffer: 38,909
Episode 250/7500 | Avg Reward: 1.78 | Best Reward: 8.00 | Epsilon: 0.8825 | Buffer: 49,147
Episode 300/7500 | Avg Reward: 1.70 | Best Reward: 8.00 | Epsilon: 0.8607 | Buffer: 59,372
Episode 350/7500 | Avg Reward: 1.36 | Best Reward: 8.00 | Epsilon: 0.8394 | Buffer: 68,773
Episode 400/7500 | Avg Reward: 2.02 | Best Reward: 8.00 | Epsilon: 0.8187 | Buffer: 79,423
```

A medida que el epsilon fue bajando y la red acumuló experiencia en el buffer, el rendimiento comenzó a escalar. Al final del entrenamiento el agente alcanzó un promedio de 22 bloques por partida con un mejor puntaje de 73, una mejora destacable respecto al inicio. El epsilon llegó al mínimo de 0.05 y el buffer se saturó en su capacidad máxima de 100,000 transiciones. 

```
Episode 7100/7500 | Avg Reward: 23.04 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7200/7500 | Avg Reward: 22.32 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7250/7500 | Avg Reward: 22.82 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7300/7500 | Avg Reward: 20.48 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7350/7500 | Avg Reward: 23.80 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7400/7500 | Avg Reward: 20.20 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7450/7500 | Avg Reward: 22.78 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
Episode 7500/7500 | Avg Reward: 22.62 | Best Reward: 73.00 | Epsilon: 0.0500 | Buffer: 100,000
```

---

## 🔍 Reflexión de los Resultados

El entrenamiento se realizó con una GPU NVIDIA RTX 3070 y tomó aproximadamente 15 horas para completar 7,500 episodios. El promedio de recompensa en los últimos bloques de 50 episodios se estabilizó alrededor de 22 puntos, con un mejor puntaje registrado de 73 en una sola partida. Para contextualizar estos resultados, según Mnih et al. (2015, p. 11), el promedio de un tester humano profesional, que jugó bajo condiciones controladas, sin pausar ni guardar partida, con aproximadamente 2 horas de práctica previa, fue de 31.8 puntos en Breakout. La DQN desplegada por los investigadores alcanzó una recompensa promedio de 401.2 con una desviación estándar de 26.9, lo que indica que su algoritmo superó ampliamente el nivel humano y jugaba de forma casi perfecta. La diferencia entre los resultados de este repositorio y los del paper se explica principalmente por la escala del entrenamiento, DeepMind entrenó su agente durante 50 millones de fotogramas (equivalente a aproximadamente 38 días de juego en tiempo real a 60 Hz), utilizó un replay buffer de 1 millón de transiciones, 10 veces más grande que el configurados de 100,000. El agente con 7,500 episodios y aproximadamente 1.5 millones de pasos, apenas ha visto una fracción de la experiencia que el agente de DeepMind acumuló, lo que sugiere que sesiones adicionales de entrenamiento y un buffer más grande podrían acercar el rendimiento a los niveles reportados en el artículo.

---

## 💭 Lo que más costó en el proyecto

El mayor desafío fue adaptar el código que ya funcionaba en otros retos de RL a las particularidades de Breakout, porque no se trataba de cambiar parámetros sino de repensar por completo cómo el agente percibe el mundo. En Breakout el estado es una imagen de 210×160 píxeles en RGB, y no existe ninguna conversión obvia que transforme eso en algo que una red densa pueda procesar eficientemente. Fue necesario investigar directamente en el paper de Mnih et al. (2015) para entender qué preprocesamiento aplicaron los investigadores de DeepMind, convertir cada fotograma a escala de grises, redimensionarlo a 84×84 y normalizarlo entre 0 y 1. Pero el paso que más costó comprender conceptualmente fue el apilamiento de fotogramas. La idea es que un solo fotograma no contiene información de movimiento, si se congela una imagen del juego, es imposible saber si la pelota va hacia arriba o hacia abajo, ni a qué velocidad se mueve. Para resolver esto, se apilan los últimos 4 fotogramas como si fueran canales de una imagen, de la misma forma en que una imagen a color tiene 3 canales (rojo, verde, azul), el estado del agente tiene 4 canales donde cada uno es un instante de tiempo distinto. Así, la CNN puede comparar las diferencias entre fotogramas consecutivos y deducir la velocidad y dirección de la pelota sin que nadie se lo diga explícitamente. Entender que esos 4 canales no son colores sino tiempo fue lo que tomó más revisión del paper y más pruebas de código hasta que funcionó correctamente.

El segundo gran desafío fue entender el ambiente antes de escribir una sola línea de código adaptado. En Breakout la estructura de recompensas es mucho más escasa, el agente solo recibe +1 cuando rompe un bloque y 0 el resto del tiempo, lo que significa que puede pasar cientos de pasos sin recibir ninguna señal de aprendizaje. Además, la acción FIRE resultó ser un detalle crítico que no era evidente al principio, sin presionar FIRE la pelota nunca se lanza, el agente se queda inmóvil y el episodio no avanza. El agente tiene que descubrir por sí mismo, a través de exploración aleatoria, que necesita ejecutar esa acción al inicio de cada vida. Comprender estas particularidades del ambiente fue fundamental para interpretar correctamente los resultados del entrenamiento y no asumir que el agente estaba fallando cuando simplemente no había aprendido aún a lanzar la pelota.

Finalmente, la transición de DQN estándar a Double DQN requirió estudiar el paper de Van Hasselt et al. (2016), que explica cómo el DQN original tiende a sobreestimar los valores Q porque la misma red que elige la mejor acción es la que evalúa su valor, un sesgo optimista que se retroalimenta con cada actualización. La solución propuesta por los autores es elegante, usar la red principal para seleccionar la mejor acción del siguiente estado, pero la red objetivo para evaluar cuánto vale esa acción. Aunque el cambio en código son apenas dos líneas, entender por qué esas dos líneas producen un entrenamiento más estable demandó comprender el problema matemático subyacente de la sobreestimación, y contrastar los resultados con el DQN estándar para verificar que efectivamente la varianza de los Q-values se reducía durante el entrenamiento.

---

## 📁 Estructura del Proyecto

```
breakout/
├── src/
│   └── rl_games/
│       ├── cli.py             # Interfaz de línea de comandos
│       └── agents/
│           └── dqn.py         # Agente DQN con CNN y PyTorch
│
├── saves/                     # Modelos guardados (generado al entrenar)
├── pyproject.toml
└── README.md
```

---

## 📦 Dependencias

| Paquete | Uso |
|---|---|
| `gymnasium[atari]` | Ambiente ALE/Breakout-v5 |
| `ale-py` | Arcade Learning Environment |
| `torch` | Red neuronal CNN para DQN |
| `torchvision` | Utilidades de visión |
| `opencv-python` | Preprocesamiento de fotogramas |
| `numpy` | Operaciones numéricas |

---

## ⚙️ Instalación

Requiere Python 3.11 y una GPU NVIDIA con CUDA 12.x (recomendado).

```bash
# Clonar el repositorio
git clone https://github.com/david76317C/rl-breakout.git
cd rl-breakou

# Instalar dependencias con uv (descarga PyTorch con soporte CUDA automáticamente)
uv sync

# Activar el entorno virtual
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / Mac
```

---

## 🕹️ Uso

### Ver información del ambiente

```bash
rlgames inspect --steps 10
```

Muestra el espacio de observaciones/acciones y ejecuta pasos con política aleatoria.

### Inicializar un agente

```bash
rlgames init dqn
```

### Entrenar

```bash
rlgames train dqn --episodes 5000
```

El entrenamiento se puede reanudar — al ejecutar `train` de nuevo, el agente carga los pesos guardados y continúa desde donde quedó con el mismo epsilon, optimizador y contadores.

### Ver información del agente guardado

```bash
rlgames load dqn
rlgames load dqn --eval     # incluye evaluación de 10 episodios
```

### Simular episodios

```bash
rlgames sim dqn --episodes 3 --steps 20
```

### Renderizar visualmente

```bash
rlgames render dqn --episodes 3
```

Abre una ventana gráfica con el agente jugando Breakout en tiempo real.

### Listar agentes disponibles

```bash
rlgames list
```

### Ver versión

```bash
rlgames version
```

---

## 📝 Notas

- El modelo entrenado se guarda en `saves/dqn_breakout.pt`.

---

## 📚 Referencias

- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533. https://www.nature.com/articles/nature14236
- Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI. https://arxiv.org/abs/1509.06461
- Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. http://www.incompleteideas.net/book/RLbook2020.pdf