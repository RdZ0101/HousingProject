import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Papa from 'papaparse';
import { getPrediction } from './salesapi';
import './SalesPred.css';

function SalesPred() {
    const [suburb, setSuburb] = useState('');
    const [postcode, setPostcode] = useState('');
    const [rooms, setRooms] = useState('');
    const [houseType, setHouseType] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [suburbToPostcodeMap, setSuburbToPostcodeMap] = useState({});
    const navigate = useNavigate();

    useEffect(() => {
        fetch(`${process.env.PUBLIC_URL}/Melbourne_housing_FULL.csv`)
            .then(response => response.text())
            .then(csvData => {
                Papa.parse(csvData, {
                    header: true,
                    complete: (results) => {
                        const mapping = {};
                        results.data.forEach(row => {
                            if (row.Suburb && row.Postcode) {
                                mapping[row.Suburb.toLowerCase()] = row.Postcode;
                            }
                        });
                        setSuburbToPostcodeMap(mapping);
                    },
                });
            });
    }, []);

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

    const mapHouseTypeToModelValue = (type) => {
        switch (type.toLowerCase()) {
            case 'house':
                return 0;
            case 'unit':
                return 1;
            case 'townhouse':
                return 2;
            default:
                return null;
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        const postcodeFromSuburb = suburbToPostcodeMap[suburb.toLowerCase()];
        
        if (!postcodeFromSuburb) {
            alert("Invalid suburb name. Please enter a valid suburb.");
            return;
        }

        setPostcode(postcodeFromSuburb);

        const houseTypeValue = mapHouseTypeToModelValue(houseType);

        if (houseTypeValue === null) {
            alert("Please select a valid house type");
            return;
        }

        const inputData = {
            postcode: postcodeFromSuburb,
            rooms: parseInt(rooms),
            house_type: houseTypeValue
        };

        const result = await getPrediction(inputData);

        if (result && result.prediction) {
            const prediction = parseFloat(result.prediction).toFixed(2);

            navigate('/GeneratedSalesPrediction', {
                state: {
                    prediction: prediction,
                    postcode: postcodeFromSuburb,
                    rooms: parseInt(rooms),
                    houseType: houseTypeValue
                }
            });
        } else {
            console.error("Failed to get prediction.");
        }
    };

    return (
        <div className="sales-prediction-container">
            <h1>Enter Details for Sales Prediction</h1>
            <form onSubmit={handleSubmit}>
                
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

                <label className="input-label">
                    Number of Rooms:
                    <select
                        value={rooms}
                        onChange={(e) => setRooms(e.target.value)}
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

                <label className="input-label">
                    Type of House:
                    <select
                        value={houseType}
                        onChange={(e) => setHouseType(e.target.value)}
                        className="input-field"
                    >
                        <option value="">Select type</option>
                        <option value="house">House</option>
                        <option value="unit">Unit</option>
                        <option value="townhouse">Townhouse</option>
                    </select>
                </label>

                <button type="submit" className="generate-button">Get Prediction</button>
            </form>
        </div>
    );
}

export default SalesPred;
