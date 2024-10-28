import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

function Footer() {
    return (
        <footer className="footer">
            <div className="footer-content">
                <div className="footer-links">
                    <Link to="/">Home</Link>
                    <span>|</span>
                    <Link to="/MarketAnalysis">Market Analysis</Link>
                    <span>|</span>
                    <Link to="/SalesPredictions">Sales Predictions</Link>
                    <span>|</span>
                    <Link to="/RentPredictions">Rent Predictions</Link>
                    <span>|</span>
                    <Link to="/AboutUs">About Us</Link>
                </div>
                <p className="footer-text">© 2024 LagOutLoud. All rights reserved.</p>
            </div>
        </footer>
    );
}

export default Footer;
