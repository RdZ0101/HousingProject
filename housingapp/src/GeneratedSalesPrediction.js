import React from 'react';
import { useLocation } from 'react-router-dom';
import PostcodePriceBarChart from './PostcodePriceBarChart';
import RoomCountLineChart from './RoomCountLineChart';

function GenSalesPred() {
    const location = useLocation();
    console.log("Location State:", location.state);
    const prediction = location.state?.prediction;
    const postcode = location.state?.postcode;
    const rooms = location.state?.rooms;
    const houseType = location.state?.houseType;

    return (
        <div style={{ textAlign: 'center', maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
            <h1>Sales Prediction</h1>

            {prediction !== undefined ? (
                <p>The predicted sales price is: ${prediction}</p>
            ) : (
                <p>No prediction available.</p>
            )}

            {postcode && rooms && houseType !== undefined ? (
                <>
                    <h2>Predicted Price Comparison by Suburb</h2>
                    <PostcodePriceBarChart 
                        userPostcode={postcode} 
                        userRooms={rooms} 
                        userHouseType={houseType} 
                    />

                    <h2>Predicted Price vs. Number of Rooms</h2>
                    <RoomCountLineChart 
                        userPostcode={postcode} 
                        houseType={houseType} 
                    />
                </>
            ) : (
                <p>Insufficient data to display charts. Please provide postcode, room count, and house type.</p>
            )}
        </div>
    );
}

export default GenSalesPred;
