"""
Aplicacion Gradio — Clasificador de Lesiones Cutaneas
RNA MLP Backpropagation — HAM10000 — UTN FRLP 2026

Uso:
    python app.py
    Luego abrir http://localhost:7860 en el navegador.
"""

import gradio as gr
import numpy as np
from PIL import Image
import joblib
import os

# Directorio donde vive este script (funciona sin importar desde donde se ejecute)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuracion de clases ──────────────────────────────────────────────────

CLASES = {
    0: {
        "codigo": "akiec",
        "nombre": "Queratosis actínica / intraepitelial",
        "riesgo": "⚠️ CONSULTAR MÉDICO",
        "nivel": "medio",
        "descripcion": (
            "Lesión potencialmente maligna. Puede evolucionar a carcinoma "
            "espinocelular. Se recomienda evaluación dermatológica."
        ),
        "color": "#f59e0b",
    },
    1: {
        "codigo": "bcc",
        "nombre": "Carcinoma basocelular",
        "riesgo": "🔴 PELIGROSO — ATENCIÓN MÉDICA URGENTE",
        "nivel": "alto",
        "descripcion": (
            "Cáncer de piel maligno. Es el tipo de cáncer más frecuente. "
            "Rara vez hace metástasis, pero requiere tratamiento inmediato."
        ),
        "color": "#ef4444",
    },
    2: {
        "codigo": "bkl",
        "nombre": "Queratosis benigna",
        "riesgo": "🟢 BENIGNO",
        "nivel": "bajo",
        "descripcion": (
            "Lesión cutánea benigna muy frecuente. No requiere tratamiento "
            "urgente, pero conviene control dermatológico periódico."
        ),
        "color": "#22c55e",
    },
    3: {
        "codigo": "df",
        "nombre": "Dermatofibroma",
        "riesgo": "🟢 BENIGNO",
        "nivel": "bajo",
        "descripcion": (
            "Tumor benigno de tejido fibroso. Generalmente no requiere "
            "tratamiento salvo por razones estéticas o si causa molestias."
        ),
        "color": "#22c55e",
    },
    4: {
        "codigo": "nv",
        "nombre": "Nevo melanocítico (lunar)",
        "riesgo": "🟢 BENIGNO",
        "nivel": "bajo",
        "descripcion": (
            "Lunar común benigno. Se recomienda control anual con dermatólogo "
            "y aplicar la regla ABCDE para detectar cambios sospechosos."
        ),
        "color": "#22c55e",
    },
    5: {
        "codigo": "vasc",
        "nombre": "Lesión vascular",
        "riesgo": "🟢 BENIGNO",
        "nivel": "bajo",
        "descripcion": (
            "Lesión de origen vascular (angioma, hemangioma). "
            "Generalmente benigna y sin riesgo de malignización."
        ),
        "color": "#22c55e",
    },
    6: {
        "codigo": "mel",
        "nombre": "Melanoma",
        "riesgo": "🔴 PELIGROSO — ATENCIÓN MÉDICA URGENTE",
        "nivel": "alto",
        "descripcion": (
            "Cáncer de piel maligno de mayor mortalidad. La detección "
            "temprana es crítica: la supervivencia a 5 años cae de 98% "
            "(estadio I) a menos del 25% en estadios avanzados."
        ),
        "color": "#ef4444",
    },
}

ADVERTENCIA_DISCLAIMER = (
    "⚕️ **AVISO MÉDICO IMPORTANTE:** Este sistema es una herramienta de "
    "asistencia académica basada en un modelo MLP entrenado con imágenes "
    "dermatoscópicas. **No reemplaza el diagnóstico de un profesional médico.** "
    "Ante cualquier lesión sospechosa, consultar a un dermatólogo."
)

ADVERTENCIA_CAMARA = (
    "📷 **Nota técnica:** El modelo fue entrenado con imágenes dermatoscópicas "
    "(obtenidas con dermatoscopio, instrumento médico especializado). "
    "Fotos de cámara convencional pueden producir predicciones menos confiables."
)

# ── Carga del modelo ─────────────────────────────────────────────────────────
# MODEL_PATH se puede sobreescribir via variable de entorno (útil en Docker).
# Por defecto usa modelo_mlp.pkl junto al script.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "modelo_mlp.pkl"),
)

def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"❌ No se encontró '{MODEL_PATH}'.\n"
            "Ejecutá primero el notebook skin_cancer_RNA.ipynb completo "
            "(incluida la celda de guardado del modelo)."
        )
    try:
        modelo = joblib.load(MODEL_PATH)
        return modelo, None
    except Exception as e:
        return None, f"❌ Error al cargar el modelo: {e}"


mlp, error_carga = cargar_modelo()

# ── Preprocesamiento ─────────────────────────────────────────────────────────

def preprocesar(imagen_np: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB (numpy array) al vector de 784 features
    que espera el modelo: escala de grises, 28×28, normalizado [0,1].
    """
    img = Image.fromarray(imagen_np.astype(np.uint8))
    img = img.convert("L")          # escala de grises
    img = img.resize((28, 28), Image.LANCZOS)  # redimensionar
    arr = np.array(img) / 255.0     # normalizar [0,1]
    return arr.flatten().reshape(1, -1)  # aplanar → (1, 784)

# ── Logica de clasificacion ──────────────────────────────────────────────────

def clasificar(imagen):
    if imagen is None:
        return (
            "Sin imagen",
            "Cargá o capturá una imagen para analizarla.",
            None,
            ADVERTENCIA_DISCLAIMER,
        )

    if mlp is None:
        return "Error", error_carga, None, ""

    # Preprocesar
    X = preprocesar(imagen)

    # Predecir probabilidades
    probs = mlp.predict_proba(X)[0]
    clase_pred = int(np.argmax(probs))
    confianza = float(probs[clase_pred]) * 100

    info = CLASES[clase_pred]

    # ── Resultado principal ──
    resultado_html = f"""
<div style="
    border-left: 6px solid {info['color']};
    background: #1a1a2e;
    padding: 16px 20px;
    border-radius: 8px;
    font-family: sans-serif;
">
    <div style="font-size: 1.4em; font-weight: bold; color: {info['color']}; margin-bottom: 6px;">
        {info['riesgo']}
    </div>
    <div style="font-size: 1.1em; color: #e2e8f0; margin-bottom: 4px;">
        <b>{info['nombre']}</b> &nbsp;
        <span style="color:#94a3b8; font-size:0.9em;">({info['codigo']})</span>
    </div>
    <div style="font-size: 0.95em; color: #94a3b8; margin-bottom: 10px;">
        Confianza del modelo: <b style="color:#e2e8f0">{confianza:.1f}%</b>
    </div>
    <div style="font-size: 0.92em; color: #cbd5e1; line-height: 1.5;">
        {info['descripcion']}
    </div>
</div>
"""

    # ── Grafico de probabilidades (dict para gr.Label) ──
    probs_dict = {
        f"{CLASES[i]['codigo']} — {CLASES[i]['nombre']}": float(probs[i])
        for i in range(7)
    }

    advertencia = f"{ADVERTENCIA_DISCLAIMER}\n\n{ADVERTENCIA_CAMARA}"

    return resultado_html, probs_dict, advertencia


# ── Interfaz Gradio ──────────────────────────────────────────────────────────

DESCRIPCION = """
# 🔬 Clasificador de Lesiones Cutáneas
### RNA MLP Backpropagation · HAM10000 · UTN FRLP 2026

Subí una foto de la lesión cutánea o usá la cámara para capturarla.
El sistema la clasificará en 7 categorías diagnósticas e indicará el nivel de riesgo.
"""

with gr.Blocks(
    title="Clasificador de Lesiones Cutáneas — UTN FRLP",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(DESCRIPCION)

    if mlp is None:
        gr.Markdown(f"## ❌ Error\n{error_carga}")
    else:
        with gr.Row():
            # ── Columna izquierda: entrada ──
            with gr.Column(scale=1):
                imagen_input = gr.Image(
                    label="Imagen de la lesión",
                    sources=["upload", "webcam", "clipboard"],
                    type="numpy",
                    height=300,
                )
                btn_analizar = gr.Button(
                    "🔍 Analizar lesión",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown(
                    "_Tip: usá 'Webcam' para capturar en tiempo real "
                    "o arrastrá una imagen directamente._"
                )

            # ── Columna derecha: resultados ──
            with gr.Column(scale=1):
                resultado_html = gr.HTML(label="Diagnóstico")
                probs_output = gr.Label(
                    label="Probabilidad por clase",
                    num_top_classes=7,
                )

        advertencia_output = gr.Markdown()

        # Evento: botón analizar
        btn_analizar.click(
            fn=clasificar,
            inputs=[imagen_input],
            outputs=[resultado_html, probs_output, advertencia_output],
        )

        # Evento: analizar automáticamente al cargar imagen
        imagen_input.change(
            fn=clasificar,
            inputs=[imagen_input],
            outputs=[resultado_html, probs_output, advertencia_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Clases del modelo:** "
            "🔴 mel (Melanoma) · "
            "🔴 bcc (Carcinoma basocelular) · "
            "🟡 akiec (Queratosis actínica) · "
            "🟢 bkl · df · nv · vasc"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # En Docker, GRADIO_LAUNCH_INBROWSER=false desactiva la apertura del navegador
    import os as _os
    _inbrowser = _os.environ.get("GRADIO_LAUNCH_INBROWSER", "true").lower() != "false"

    demo.launch(
        server_name=_os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(_os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,        # True para generar link publico temporal
        inbrowser=_inbrowser,
    )
