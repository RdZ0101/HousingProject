import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './SalesPred.css';
import './GeneratedSalesPrediction';

function SalesPred() {
    const [suburb, setSuburb] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [numRooms, setNumRooms] = useState('');
    const [houseType, setHouseType] = useState('');

    const handleSuburbChange = async (e) => {
        const userInput = e.target.value;
        setSuburb(userInput);

        if (userInput) {
            const response = await fetch(`http://localhost:5000/api/suburbs?query=${userInput}`);
            const data = await response.json();
            setSuggestions(data);
        } else {
            setSuggestions([]);
        }
    };

    return (
        <div className="sales-prediction-container">
            <h1>View the sales price of the house you are looking for</h1>

            {/* Suburb Input */}
            <label className="input-label">
                Suburb: 
                <input 
                    type="text" 
                    value={suburb} 
                    onChange={handleSuburbChange} 
                    placeholder="Enter suburb"
                    autoComplete="off"
                    className="input-field"
                />
            </label>
            {suggestions.length > 0 && (
                <ul className="suggestions-list">
                    {suggestions.map((suggestion, index) => (
                        <li 
                            key={index}
                            onClick={() => {
                                setSuburb(suggestion);
                                setSuggestions([]);
                            }}
                            className="suggestion-item"
                        >
                            {suggestion}
                        </li>
                    ))}
                </ul>
            )}

            {/* Number of Rooms Dropdown */}
            <label className="input-label">
                Number of rooms:
                <select 
                    value={numRooms} 
                    onChange={(e) => setNumRooms(e.target.value)}
                    className="input-field"
                >
                    <option value="">Select number of rooms</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                    <option value="6">6</option>
                    <option value="7">7</option>
                </select>
            </label>

            {/* Type of House Dropdown */}
            <label className="input-label">
                Type of house:
                <select 
                    value={houseType} 
                    onChange={(e) => setHouseType(e.target.value)}
                    className="input-field"
                >
                    <option value="">Select house type</option>
                    <option value="House">House</option>
                    <option value="Townhouse">Townhouse</option>
                    <option value="Unit">Unit</option>
                </select>
            </label>

            {/* Button to Navigate to Generated Sales Prediction */}
            <Link to="/GeneratedSalesPrediction">
                <button className="generate-button">Generate Sales Prediction</button>
            </Link>
        </div>
    );
}

export default SalesPred;
