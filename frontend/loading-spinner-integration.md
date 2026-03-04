# Integración del Componente de Carga (Spinner)

El `KaudalLoadingSpinner` está diseñado para ser completamente agnóstico al framework y funcionar utilizando clases de Vanilla JS.

Dado que tu API de FastAPI usa Server-Sent Events (SSE) para el endpoint `/api/optimizar`, este componente se encarga de escuchar los eventos del flujo (stream), actualizar la barra de progreso, mostrar los mensajes rotativos educativos y manejar errores, todo sin bloquear el frontend de manera forzosa ni depender de librerías extra.

## Archivos que proveen la funcionalidad
1. `frontend/loading-spinner.css`: Contiene la animación de CSS (el anillo gradiente estelar de Kaudal), diseño "glassmorphism", dark mode default y display de estados flexibles (Error, Timeout).
2. `frontend/loading-spinner.js`: Exporta `KaudalLoadingSpinner`, la clase que se auto-inyecta en el DOM al instanciarse.

## Cómo implementarlo en la UI actual

Para integrarlo, sigue este bloque de código dentro de la lógica de tu formulario (ej. al presionar el botón "Optimizar"):

```html
<!-- En tu HTML principal (index.html o dashboard.html) añade los links: -->
<link rel="stylesheet" href="loading-spinner.css">
<script src="loading-spinner.js"></script>
```

```javascript
// En tu javascript de lógica, ejemplo:
const btnOpt = document.getElementById("btn-optimizar");

btnOpt.addEventListener("click", async () => {
    // 1. Instanciar el Spinner
    const spinner = new KaudalLoadingSpinner();
    
    // Configurar payload
    const payload = {
        tickers: ["AAPL", "MSFT", "CEMEXCPO.MX"],
        budget: 15000,
        period: "3y"
    };

    // 2. Ejecutar Fetch con conexión SSE envuelta
    try {
        const finalResult = await spinner.fetchWithSSE("/api/optimizar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                // "Authorization": "Bearer " + token // <- Si manejas Supabase Session Manager
            },
            body: JSON.stringify(payload)
        });

        // 3. Manejar respuesta
        if (finalResult) {
            console.log("¡Optimización Exitosa!", finalResult);
            // Mostrar resultados en tu UI (Gráficas, Tablas, etc.)
            // El spinner se cierra automáticamente si fue exitoso (100%).
        }
    } catch (e) {
        // En caso de catch duro (conexión caída pura), la UI se quedó en spinner de error.
        console.error("Fallo de red total", e);
    }
});
```

### Características Automáticas de este Spinner
* **Rotación de Mensajes**: Cada 3 segundos cambiará entre una bolsa de 25 mensajes balanceados en 5 categorías (Tips, Curiosidades, Históricos, Frases y Estado actual).
* **SSE Progression**: La barra avanza desde el 20% al descargar, 40% al limpiar, 60% al calcular covarianza, 80% optimizando, hasta el 100% "done".
* **Timeout & Error Handling**: A los 15 segundos permite cancelar, o bien detecta los `{"stage": "error"}` provenientes del backend emitiendo el componente visual con ícono triste y listando los tickers defectuosos.
