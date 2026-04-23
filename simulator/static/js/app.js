document.addEventListener('DOMContentLoaded', () => {
    
    // DOM Elements
    const valRpm = document.getElementById('val-rpm');
    const barRpm = document.getElementById('bar-rpm');
    
    const valLoad = document.getElementById('val-load');
    const barLoad = document.getElementById('bar-load');
    
    const valSpeed = document.getElementById('val-speed');
    const barSpeed = document.getElementById('bar-speed');
    
    const valLatency = document.getElementById('val-latency');
    const valAnomaly = document.getElementById('val-anomaly');
    const alertBanner = document.getElementById('alert-banner');
    
    // Polling Loop
    setInterval(fetchTelemetry, 1000);

    async function fetchTelemetry() {
        try {
            const response = await fetch('/stream');
            const data = await response.json();
            
            updateDashboard(data);
        } catch (error) {
            console.error('Error fetching telemetry:', error);
        }
    }

    function updateDashboard(data) {
        // Animate numbers (simple approach)
        valRpm.innerText = data.rpm.toFixed(0);
        valLoad.innerText = data.load.toFixed(1);
        valSpeed.innerText = data.speed.toFixed(0);
        valLatency.innerText = data.latency_ms.toFixed(2) + ' ms';
        valAnomaly.innerText = data.anomaly_score.toFixed(4);
        
        // Update bars
        // Max RPM ~7000
        const rpmPct = Math.min((data.rpm / 7000) * 100, 100);
        barRpm.style.width = rpmPct + '%';
        
        // Max Load ~100
        barLoad.style.width = data.load + '%';
        
        // Max Speed ~200
        const speedPct = Math.min((data.speed / 200) * 100, 100);
        barSpeed.style.width = speedPct + '%';
        
        // Handle Prediction
        if (data.prediction === 1) {
            document.body.classList.add('danger-state');
            alertBanner.classList.remove('hidden');
        } else {
            document.body.classList.remove('danger-state');
            alertBanner.classList.add('hidden');
        }
    }
});
