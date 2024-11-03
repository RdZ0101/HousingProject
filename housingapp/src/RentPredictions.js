import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './SalesPred.css';
import './GeneratedRentPredictions';

function RentPred() {
    const [suburb, setSuburb] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [numRooms, setNumRooms] = useState('');
    const [houseType, setHouseType] = useState('');
    const [rentalPeriod, setRentalPeriod] = useState('');

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
        const value = Math.max(1, parseInt(e.target.value) || 1);
        setRentalPeriod(value);
    };

    // Update the number of rooms options based on the selected house type
    const handleHouseTypeChange = (e) => {
        const selectedType = e.target.value;
        setHouseType(selectedType);
        
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

<<<<<<< HEAD
    /*
    const rentPredictionRequest = async () => {
        if (!suburb || !numRooms || !houseType || !rentalPeriod) {
            alert("Please fill out all fields");
            return;
        }
        
        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ suburb, numRooms, houseType, rentalPeriod }),
            });

            if (response.ok) {
                const predictionData = await response.json();
                // Pass predictionData as state to the prediction page
                navigate('/GeneratedRentPredictions', { state: { prediction: predictionData } });
            } else {
                alert("Failed to fetch prediction. Please try again.");
            }
        } catch (error) {
            console.error("Error fetching prediction:", error);
            alert("An error occurred. Please try again.");
        }
    };*/




=======
    // Function to send the rent prediction request
    const rentPredictionRequest = async () => {
        if (!suburb || !houseType || !numRooms || !rentalPeriod) {
            alert('Please fill in all fields');
            return;
        }

        const requestData = {
            suburb: suburb,
            numRooms: parseInt(numRooms),
            houseType: houseType,
            rentalPeriod: parseInt(rentalPeriod)
        };

        try {
            const response = await fetch("http://0.0.0.0:8000/predict_rent", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            console.log("Predicted Rent:", data.predicted_price);
            // Handle the predicted price (e.g., display it on the page)
        } catch (error) {
            console.error("Error fetching prediction:", error);
        }
    };
>>>>>>> 42f15d95aa5510390181c2adb83266e93193aade

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

export default RentPred;


