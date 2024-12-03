import React from "react";
import { Link } from "react-router-dom";
import './NavBar.css';

function NavBar() {
    return (
        <nav className="navbar">
            <ul className="nav-list">
                <li><Link to="/">Home</Link></li>
                <li><Link to="/MarketAnalysis">Market Analysis</Link></li>
                <li><Link to="/SalesPredictions">Sales Predictions</Link></li>
                <li><Link to="/RentPredictions">Rent Predictions</Link></li>
            </ul>
        </nav>
    );
}

export default NavBar;
