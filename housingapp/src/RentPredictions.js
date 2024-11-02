import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './SalesPred.css';
import './GeneratedRentPredictions';

function RentPred() {
    const [suburb, setSuburb] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [numRooms, setNumRooms] = useState('');
    const [houseType, setHouseType] = useState('');
    const [rentalPeriod, setRentalPeriod] = useState(''); // New state for rental period

    const handleSuburbChange = async (e) => {
        const userInput = e.target.value;
        setSuburb(userInput);

        if (userInput) {
            const response = await fetch(`http://localhost:5001/api/suburbs?query=${userInput}`);
            const data = await response.json();
            setSuggestions(data);
        } else {
            setSuggestions([]);
        }
    };

    // Handler to ensure rentalPeriod stays non-negative
    const handleRentalPeriodChange = (e) => {
        const value = Math.max(1, parseInt(e.target.value) || 1); // Ensures minimum value of 1
        setRentalPeriod(value);
    };

    // Update the number of rooms options based on the selected house type
    const handleHouseTypeChange = (e) => {
        const selectedType = e.target.value;
        setHouseType(selectedType);
        
        // Reset numRooms if it is not a valid option for the selected house type
        if ((selectedType === 'Flat' && !['1', '2', '3'].includes(numRooms)) || 
            (selectedType === 'House' && !['2', '3', '4'].includes(numRooms))) {
            setNumRooms('');
        }
    };

    // Determine options for number of rooms based on house type
    const getRoomOptions = () => {
        if (houseType === 'Flat') {
            return ['1', '2', '3'];
        } else if (houseType === 'House') {
            return ['2', '3', '4'];
        }
        return [];
    };

    return (
        <div className="sales-prediction-container">
            <h1>View the rent price of the house you are looking for</h1>

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

            {/* Type of House Dropdown */}
            <label className="input-label">
                Type of house:
                <select 
                    value={houseType} 
                    onChange={handleHouseTypeChange}
                    className="input-field"
                >
                    <option value="">Select house type</option>
                    <option value="House">House</option>
                    <option value="Flat">Flat</option>
                </select>
            </label>

            {/* Number of Rooms Dropdown */}
            <label className="input-label">
                Number of rooms:
                <select 
                    value={numRooms} 
                    onChange={(e) => setNumRooms(e.target.value)}
                    className="input-field"
                >
                    <option value="">Select number of rooms</option>
                    {getRoomOptions().map((room) => (
                        <option key={room} value={room}>{room}</option>
                    ))}
                </select>
            </label>

            {/* Time Input */}
            <label className="input-label">
                I want to see the rental price in: 
                <input 
                    type="number" 
                    value={rentalPeriod} 
                    onChange={handleRentalPeriodChange}
                    placeholder="Enter months"
                    className="input-field"
                    min="1" // Prevents values less than 1 in most browsers
                />
            </label>

            {/* Button to Navigate to Generated Rent Prediction */}
            <Link to="/GeneratedRentPredictions">
                <button className="generate-button">Generate Rent Prediction</button>
            </Link>
        </div>
    );
}

export default RentPred;
