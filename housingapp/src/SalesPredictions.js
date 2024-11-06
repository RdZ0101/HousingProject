import React, { useState, useEffect } from 'react';
import { getPrediction } from './salesapi';
import { useNavigate } from 'react-router-dom';
import Papa from 'papaparse';

function SalesPred() {
    const [suburb, setSuburb] = useState('');
    const [postcode, setPostcode] = useState('');
    const [rooms, setRooms] = useState('');
    const [houseType, setHouseType] = useState('');
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

    const mapHouseTypeToModelValue = (type) => {
        switch (type) {
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

            navigate('/generated-sales-prediction', {
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
        <div>
            <h1>Enter Details for Sales Prediction</h1>
            <form onSubmit={handleSubmit}>
                <label>
                    Enter Suburb:
                    <input
                        type="text"
                        value={suburb}
                        onChange={(e) => setSuburb(e.target.value)}
                    />
                </label>
                <br />
                <label>
                    Enter Number of Rooms:
                    <input
                        type="number"
                        value={rooms}
                        onChange={(e) => setRooms(e.target.value)}
                    />
                </label>
                <br />
                <label>
                    Select House Type:
                    <select
                        value={houseType}
                        onChange={(e) => setHouseType(e.target.value)}
                    >
                        <option value="">Select Type</option>
                        <option value="house">House</option>
                        <option value="unit">Unit</option>
                        <option value="townhouse">Townhouse</option>
                    </select>
                </label>
                <br />
                <button type="submit">Get Prediction</button>
            </form>
        </div>
    );
}

export default SalesPred;
