import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';


function Home() {
    return (
        <div className="home-container">
            <div className="banner">
                <h1>Looking for the perfect home in Victoria?</h1>
                <p>Find out which suburb is just right for you</p>
            </div>
            <div className="options">
                <Link to="/SalesPredictions" className="option buy">
                    <div className="overlay">
                        <h2>Buy</h2>
                    </div>
                </Link>
                <Link to="/RentPredictions" className="option rent">
                    <div className="overlay">
                        <h2>Rent</h2>
                    </div>
                </Link>
            </div>
        </div>
    );
}

export default Home;
