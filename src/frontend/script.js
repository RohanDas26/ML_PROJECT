const configForm = document.getElementById('predict-form');
const buttonSelect = document.getElementById('submit-btn');
const buttonTextStr = document.querySelector('.btn-text');
const loadingSpinConfig = document.getElementById('loading-spinner');
const metricGrids = document.getElementById('metrics-grid');

let forecastChartObject = null;

// Ensure chart scales using custom styles corresponding to the CSS
Chart.defaults.color = "#a4b0be";
Chart.defaults.font.family = "'Inter', sans-serif";

configForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI Loading state
    buttonSelect.style.pointerEvents = "none";
    buttonTextStr.textContent = "Processing Features...";
    loadingSpinConfig.classList.remove('hidden');

    const formData = new FormData(configForm);

    const payload = {
        sector: formData.get('sector'),
        horizon: parseInt(formData.get('horizon')),
        useFeatures: document.getElementById('feature').checked,
    };

    try {
        // Normally this would be something like:
        // const res = await fetch('/api/predict', { method: 'POST', ... });
        // const data = await res.json();

        // Simulating ML Backend Delay Analysis for demo cases
        await new Promise(r => setTimeout(r, 1800));
        buttonTextStr.textContent = "Running Predictor pipeline...";
        await new Promise(r => setTimeout(r, 1500));

        const syntheticData = generateForecastPaths(payload.horizon, payload.sector);

        renderSystemPanels(syntheticData, payload.sector);

    } catch (error) {
        console.error("API call rejected or model error:", error);
        alert("Model inference failure. Check server logs!");
    } finally {
        buttonSelect.style.pointerEvents = "auto";
        buttonTextStr.textContent = "Generate Forecast";
        loadingSpinConfig.classList.add('hidden');
    }
});


function renderSystemPanels(data, sector) {
    // 1. Populate table blocks
    metricGrids.innerHTML = '';
    data.forEach(item => {
        const dStr = new Date(item.date).toLocaleDateString(undefined, { month: 'short', year: 'numeric' });

        const card = document.createElement('div');
        card.className = "metric-card";
        card.innerHTML = `
            <div class="metric-date">${dStr}</div>
            <div class="metric-value">${item.value.toFixed(2)}</div>
        `;
        metricGrids.appendChild(card);
    });

    // 2. Render smooth line charts
    const ctx = document.getElementById('forecastChart').getContext('2d');

    if (forecastChartObject) {
        forecastChartObject.destroy();
    }

    const grad = ctx.createLinearGradient(0, 0, 0, 400);
    grad.addColorStop(0, "rgba(0, 210, 211, 0.4)");
    grad.addColorStop(1, "rgba(0, 210, 211, 0.0)");

    forecastChartObject = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => new Date(d.date).toLocaleDateString(undefined, { month: 'short', year: 'numeric' })),
            datasets: [{
                label: `${sector} Energy (Trillion BTU)`,
                data: data.map(d => d.value),
                borderColor: '#00d2d3',
                backgroundColor: grad,
                borderWidth: 3,
                pointBackgroundColor: '#fff',
                pointBorderColor: '#00d2d3',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end'
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 18, 27, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#00d2d3',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: (context) => context.parsed.y.toFixed(2) + " Trillion BTU"
                    }
                }
            },
            scales: {
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        padding: 10
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        padding: 10,
                        maxTicksLimit: 12
                    }
                }
            }
        }
    });
}

// Emulate backend outputs based on sector
function generateForecastPaths(horizon, sector) {
    const dates = [];
    const baseDate = new Date();
    baseDate.setMonth(baseDate.getMonth() + 1);

    let baseline = 300;
    let seasonalAmp = 50;

    if (sector === 'Commercial') {
        baseline = 400; seasonalAmp = 100;
    } else if (sector === 'Residential') {
        baseline = 1800; seasonalAmp = 500;
    } else if (sector === 'Industrial') {
        baseline = 2500; seasonalAmp = 40;
    } else {
        baseline = 2200; seasonalAmp = 80;
    }

    const output = [];
    for (let i = 0; i < horizon; i++) {
        const m = (baseDate.getMonth() + i) % 12;
        // Basic sin wave for fake prediction curve simulation
        const isWinter = (m === 11 || m === 0 || m === 1);
        const seasonality = Math.sin((m / 11) * Math.PI * 2) * seasonalAmp;

        const noise = (Math.random() * 20) - 10;
        const trend = (i * 1.5);

        let val = baseline + seasonality + trend + noise;

        output.push({
            date: new Date(baseDate.getFullYear(), baseDate.getMonth() + i, 1).toISOString(),
            value: val
        });
    }
    return output;
}
