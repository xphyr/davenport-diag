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
    
    // Control Elements
    const btnGas = document.getElementById('btn-gas');
    const btnBrake = document.getElementById('btn-brake');
    const sliderLoad = document.getElementById('slider-load');
    const loadValue = document.getElementById('load-value');
    
    // Control State
    let controls = {
        throttle: 0.0,
        brake: 0.0,
        ext_load: 20.0
    };
    
    // Send controls to backend
    const sendControls = async () => {
        console.log('Attempting to send controls:', controls);
        try {
            const response = await fetch('/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(controls)
            });
            console.log('Control response status:', response.status);
        } catch (e) {
            console.error('Failed to send controls', e);
        }
    };
    
    // Pedal Events
    const setThrottle = (val) => { 
        if (controls.throttle === val) return;
        console.log('Setting throttle:', val);
        controls.throttle = val; 
        sendControls(); 
    };
    const setBrake = (val) => { 
        if (controls.brake === val) return;
        console.log('Setting brake:', val);
        controls.brake = val; 
        sendControls(); 
    };
    
    if (btnGas) {
        btnGas.addEventListener('mousedown', () => setThrottle(1.0));
        btnGas.addEventListener('touchstart', (e) => { e.preventDefault(); setThrottle(1.0); });
        
        ['mouseup', 'mouseleave', 'touchend'].forEach(evt => {
            btnGas.addEventListener(evt, () => setThrottle(0.0));
        });
    }
    
    if (btnBrake) {
        btnBrake.addEventListener('mousedown', () => setBrake(1.0));
        btnBrake.addEventListener('touchstart', (e) => { e.preventDefault(); setBrake(1.0); });
        
        ['mouseup', 'mouseleave', 'touchend'].forEach(evt => {
            btnBrake.addEventListener(evt, () => setBrake(0.0));
        });
    }
    
    if (sliderLoad) {
        sliderLoad.oninput = (e) => {
            controls.ext_load = parseFloat(e.target.value);
            if (loadValue) loadValue.innerText = controls.ext_load + '%';
            sendControls();
        };
    }
    
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
        if (!data) return;
        
        // Update values with null checks
        if (valRpm) valRpm.innerText = data.rpm.toFixed(0);
        if (valLoad) valLoad.innerText = data.load.toFixed(1);
        if (valSpeed) valSpeed.innerText = data.speed.toFixed(0);
        if (valLatency) valLatency.innerText = (data.latency_ms || 0).toFixed(2) + ' ms';
        if (valAnomaly) valAnomaly.innerText = (data.anomaly_score || 0).toFixed(4);
        
        // Update bars
        if (barRpm) {
            const rpmPct = Math.min((data.rpm / 7000) * 100, 100);
            barRpm.style.width = rpmPct + '%';
        }
        
        if (barLoad) barLoad.style.width = data.load + '%';
        
        if (barSpeed) {
            const speedPct = Math.min((data.speed / 200) * 100, 100);
            barSpeed.style.width = speedPct + '%';
        }
        
        // Handle Prediction
        if (data.prediction === 1) {
            document.body.classList.add('danger-state');
            if (alertBanner) alertBanner.classList.remove('hidden');
        } else {
            document.body.classList.remove('danger-state');
            if (alertBanner) alertBanner.classList.add('hidden');
        }
    }
});
