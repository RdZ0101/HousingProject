import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import './Prediction.css';

const GeneratedRentPrediction = () => {
    const location = useLocation();
    const { suburb, houseType, numRooms, rentalPeriod, predictedPrice } = location.state || {};
    const normalizedSuburb = suburb
        .toLowerCase()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    const [historicalPrices, setHistoricalPrices] = useState(null);

    useEffect(() => {
        const fetchHistoricalPrices = async () => {
            try {
                const response = await fetch('http://localhost:8000/getHistoricalRentPrices', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        suburb: suburb,
                        numRooms: numRooms,
                        houseType: houseType,
                        monthsBack: 6,
                    }),
                });
                
                if (!response.ok) {
                    throw new Error("Failed to fetch historical rent prices");
                }
                const data = await response.json();
                setHistoricalPrices(data.historical_data);
            } catch (error) {
                console.error("Error fetching historical rent prices:", error);
            }
        };

        fetchHistoricalPrices();
    }, [suburb, numRooms, houseType]);

    return (
        <div className='prediction-container'>
            <h2>
                Generated rent predictions for a {numRooms} bedroom {houseType} in {normalizedSuburb} for the next {rentalPeriod} months
            </h2>
            <h3>Average rent per week would be {predictedPrice} AUD</h3>
            {historicalPrices ? (
                <div>
                    <h4>Historical Rent Prices (last 6 months):</h4>
                    <ul>
                        {historicalPrices.map((entry, index) => (
                            <li key={index}>
                                Date: {entry.date}, Price: {entry.price} AUD
                            </li>
                        ))}
                    </ul>
                </div>
            ) : (
                <p>Loading historical prices...</p>
            )}
        </div>
    );
};

export default GeneratedRentPrediction;
