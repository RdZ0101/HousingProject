// RentPrediction.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './SalesPred.css';

function RentPrediction() {
    const [suburb, setSuburb] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [numRooms, setNumRooms] = useState('');
    const [houseType, setHouseType] = useState('');
    const [rentalPeriod, setRentalPeriod] = useState('');
    const navigate = useNavigate();

    const handleSuburbChange = async (e) => {
        const userInput = e.target.value;
        setSuburb(userInput);

        if (userInput) {
            try {
                const response = await fetch(`http://localhost:5001/api/suburbs?query=${userInput}`);
                if (response.ok) {
                    const data = await response.json();
                    setSuggestions(data);
                } else {
                    console.error("Failed to fetch suburb suggestions");
                }
            } catch (error) {
                console.error("Error fetching suggestions:", error);
            }
        } else {
            setSuggestions([]);
        }
    };

    const handleRentalPeriodChange = (e) => {
        const value = Math.max(1, parseInt(e.target.value) || 1);
        setRentalPeriod(value);
    };

    const handleHouseTypeChange = (e) => {
        const selectedType = e.target.value;
        setHouseType(selectedType);

        if ((selectedType === 'Flat' && !['1', '2', '3'].includes(numRooms)) || 
            (selectedType === 'House' && !['2', '3', '4'].includes(numRooms))) {
            setNumRooms('');
        }
    };

    const getRoomOptions = () => {
        if (houseType === 'Flat') {
            return ['1', '2', '3'];
        } else if (houseType === 'House') {
            return ['2', '3', '4'];
        }
        return [];
    };

    const rentPredictionRequest = async () => {
        if (!suburb || !houseType || !numRooms || !rentalPeriod) {
            alert('Please fill in all fields');
            return;
        }

        const requestData = {
            suburb,
            numRooms: parseInt(numRooms),
            houseType,
            rentalPeriod: parseInt(rentalPeriod)
        };

        try {
            const response = await fetch("http://localhost:8000/predict_rent", { 
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });
            if (response.ok) {
                const data = await response.json();
                // Navigate to GeneratedRentPrediction with the prediction result
                navigate("/GeneratedRentPredictions", { 
                    state: { 
                        predictedPrice: data.predicted_price,
                        suburb,
                        houseType,
                        numRooms,
                        rentalPeriod
                    } 
                });
            } else {
                console.error("Error fetching prediction");
            }
        } catch (error) {
            console.error("Error fetching prediction:", error);
        }
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

            {/* Rental Period Input */}
            <label className="input-label">
                I want to see the rental price in: 
                <input 
                    type="number" 
                    value={rentalPeriod} 
                    onChange={handleRentalPeriodChange}
                    placeholder="Enter months"
                    className="input-field"
                    min="1"
                />
            </label>

            {/* Button to Generate Rent Prediction */}
            <button className="generate-button" onClick={rentPredictionRequest}>Generate Rent Prediction</button>
        </div>
    );
}

export default RentPrediction;

