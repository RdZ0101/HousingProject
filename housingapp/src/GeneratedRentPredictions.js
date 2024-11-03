import React, { useEffect, useState, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import * as d3 from 'd3';
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
    const svgRef = useRef(); // Reference for D3 chart

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

    // Set up D3 chart after data is fetched
    useEffect(() => {
        if (historicalPrices) {
            // Define dimensions and margins
            const width = 600;
            const height = 400;
            const margin = { top: 20, right: 30, bottom: 70, left: 60 };

            // Parse dates and format data for D3
            const parseDate = d3.timeParse('%m-%Y');
            const data = historicalPrices
                .map(d => ({
                    date: parseDate(d.date), // Parses dates like "12-2022" to Date objects
                    price: +d.price,         // Converts price to a number
                }))
                .filter(d => d.date); // Filter out invalid dates

            // Define scales
            const xScale = d3.scaleTime()
                .domain(d3.extent(data, d => d.date))
                .range([margin.left, width - margin.right]);

            const yScale = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.price)]).nice()
                .range([height - margin.bottom, margin.top]);

            // Create the line generator
            const line = d3.line()
                .x(d => xScale(d.date))
                .y(d => yScale(d.price));

            // Clear previous svg content if any
            d3.select(svgRef.current).selectAll('*').remove();

            // Create the SVG container
            const svg = d3.select(svgRef.current)
                .attr('width', width)
                .attr('height', height);

            // Add X axis
            svg.append('g')
                .attr('transform', `translate(0,${height - margin.bottom})`)
                .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%b %Y')))
                .selectAll("text")
                .attr("transform", "rotate(-45)")
                .style("text-anchor", "end");

            // Add Y axis
            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(yScale));

            // Add line path
            svg.append('path')
                .datum(data)
                .attr('fill', 'none')
                .attr('stroke', 'steelblue')
                .attr('stroke-width', 2)
                .attr('d', line);
            
            // Add points to the line
            svg.selectAll("circle")
                .data(data)
                .join("circle")
                .attr("cx", d => xScale(d.date))
                .attr("cy", d => yScale(d.price))
                .attr("r", 4)
                .attr("fill", "steelblue");
        }
    }, [historicalPrices]);

    return (
        <div className='prediction-container'>
            <h2>
                Generated rent predictions for a {numRooms} bedroom {houseType} in {normalizedSuburb} for the next {rentalPeriod} months
            </h2>
            <h3>Average rent per week would be {predictedPrice} AUD</h3>
            <div className='historical-price-chart'>
                <h4>Historical Rent Prices (last 6 months):</h4>
                <svg ref={svgRef}></svg>
            </div>
        </div>
    );
};

export default GeneratedRentPrediction;
