import React from 'react';
import './Legend.css';

function Legend() {
    return (
        <div className="legend">
            <h3>Legend</h3>
            <div className="legend-item">
                <div className="legend-color red"></div>
                <span>Rent price is greater than AUD 400</span>
            </div>
            <div className="legend-item">
                <div className="legend-color orange"></div>
                <span>Rent price is greater than AUD 300</span>
            </div>
            <div className="legend-item">
                <div className="legend-color green"></div>
                <span>Rent price is greater than AUD 200</span>
            </div>
        </div>
    );
}

export default Legend;
