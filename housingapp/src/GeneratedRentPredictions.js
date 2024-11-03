// GeneratedRentPrediction.js
import React from 'react';
import { useLocation } from 'react-router-dom';
import './Prediction.css';

const GeneratedRentPrediction = () => {
    const location = useLocation();
    const { suburb, houseType, numRooms, rentalPeriod, predictedPrice } = location.state || {};
    const normalizedSuburb = suburb
    .toLowerCase()                 // Convert the entire string to lowercase first
    .split(' ')                    // Split by spaces to get each word
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))  // Capitalize first letter of each word
    .join(' ');                    // Join words back with spaces

    return (
        <div className='prediction-container'>
            <h2>Generated rent predictions for a {numRooms} bedroom {houseType} in {normalizedSuburb} for the next {rentalPeriod} months</h2>
            <h3>Average rent per week would be {predictedPrice} AUD</h3>
        </div>
    );
};

export default GeneratedRentPrediction;
