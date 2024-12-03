import React, { useEffect, useState, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import * as d3 from 'd3';
import './RentPred.css';

const GeneratedRentPrediction = () => {
    const location = useLocation();
    const { suburb, houseType, numRooms, rentalPeriod, predictedPrice } = location.state || {};
    const normalizedSuburb = suburb
        .toLowerCase()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    const [historicalPrices, setHistoricalPrices] = useState(null);
    const [rentComparisons, setRentComparisons] = useState(null);
    const [loading, setLoading] = useState(true); // Track loading state
    const svgRef = useRef();
    const comparisonSvgRef = useRef();

    useEffect(() => {
        const fetchHistoricalPrices = async () => {
            try {
                const response = await fetch('http://localhost:8000/get_historical_rent_prices', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ suburb, numRooms, houseType, monthsBack: 6 }),
                });
                if (!response.ok) throw new Error("Failed to fetch historical rent prices");
                const data = await response.json();
                setHistoricalPrices(data.historical_data);
            } catch (error) {
                console.error("Error fetching historical rent prices:", error);
            }
        };

        const fetchRentComparisson = async () => {
            try {
                const response = await fetch('http://localhost:8000/get_rent_comparisson', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ suburb, numRooms, houseType, monthsAhead: rentalPeriod }),
                });
                if (!response.ok) throw new Error("Failed to fetch rent comparison data");
                const data = await response.json();
                setRentComparisons(data.predictions);
            } catch (error) {
                console.error("Error fetching rent comparison data:", error);
            }
        };

        // Fetch both datasets, then stop showing loading spinner
        Promise.all([fetchHistoricalPrices(), fetchRentComparisson()]).then(() => setLoading(false));
    }, [suburb, numRooms, houseType, rentalPeriod]);

    useEffect(() => {
        if (historicalPrices && !loading) { // Ensure data is loaded and page is not in loading state
            const width = 600;
            const height = 400;
            const margin = { top: 20, right: 30, bottom: 70, left: 60 };
    
            const parseDate = d3.timeParse('%m-%Y');
            const data = historicalPrices
                .map(d => ({
                    date: parseDate(d.date), // Parse the date
                    price: +d.price,         // Ensure price is a number
                }))
                .filter(d => d.date);       // Remove any invalid dates
    
            const xScale = d3.scaleTime()
                .domain(d3.extent(data, d => d.date))
                .range([margin.left, width - margin.right]);
    
            const yScale = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.price)]).nice()
                .range([height - margin.bottom, margin.top]);
    
            const line = d3.line()
                .x(d => xScale(d.date))
                .y(d => yScale(d.price));
    
            d3.select(svgRef.current).selectAll('*').remove();
    
            const svg = d3.select(svgRef.current)
                .attr('width', width)
                .attr('height', height);
    
            // X axis
            svg.append('g')
                .attr('transform', `translate(0,${height - margin.bottom})`)
                .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%b %Y')))
                .selectAll("text")
                .attr("transform", "rotate(-45)")
                .attr("y", 10)
                .style("text-anchor", "end");
    
            // X-axis label
            svg.append("text")
                .attr("text-anchor", "middle")
                .attr("x", width / 2)
                .attr("y", height - margin.bottom + 70)
                .attr("fill", "black")
                .text("Date");
    
            // Y axis
            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(yScale));
    
            // Y-axis label
            svg.append("text")
                .attr("text-anchor", "middle")
                .attr("transform", `rotate(-90)`)
                .attr("x", -height / 2)
                .attr("y", margin.left - 40)
                .attr("fill", "black")
                .text("Price per week (AUD)");
    
            // Line path
            svg.append('path')
                .datum(data)
                .attr('fill', 'none')
                .attr('stroke', 'steelblue')
                .attr('stroke-width', 2)
                .attr('d', line);
    
            // Tooltip setup
            const tooltip = d3.select(".prediction-container")
                .append("div")
                .attr("class", "tooltip")
                .style("opacity", 0)
                .style("position", "absolute")
                .style("background-color", "#fff")
                .style("border", "1px solid #d3d3d3")
                .style("padding", "5px")
                .style("border-radius", "5px")
                .style("pointer-events", "none");
    
            // Points and hover interactions
            svg.selectAll("circle")
                .data(data)
                .join("circle")
                .attr("cx", d => xScale(d.date))
                .attr("cy", d => yScale(d.price))
                .attr("r", 4)
                .attr("fill", "steelblue")
                .on("mouseover", (event, d) => {
                    tooltip.transition().duration(200).style("opacity", 0.9);
                    tooltip.html(`Price: AUD ${d.price.toFixed(2)}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 20) + "px");
                })
                .on("mouseout", () => {
                    tooltip.transition().duration(500).style("opacity", 0);
                })
                .on("mousemove", (event) => {
                    tooltip.style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 20) + "px");
                });
        }
    }, [historicalPrices, loading]);
    



    // Set up grouped bar chart for rent comparison data
    useEffect(() => {
        if (rentComparisons) {
            const width = 600;
            const height = 400;
            const margin = { top: 20, right: 150, bottom: 100, left: 60 };

            const groupedData = d3.groups(rentComparisons, d => d.suburb);

            const x0 = d3.scaleBand()
                .domain(groupedData.map(d => d[0]))
                .range([margin.left, width - margin.right])
                .padding(0.2);

            const x1 = d3.scaleBand()
                .domain(["Flat", "House"])
                .range([0, x0.bandwidth()])
                .padding(0.05);

            const yScale = d3.scaleLinear()
                .domain([0, d3.max(rentComparisons, d => d.price)]).nice()
                .range([height - margin.bottom, margin.top]);

            const colorScale = d3.scaleOrdinal()
                .domain(["F", "H"])
                .range(["#1f77b4", "#ff7f0e"]);

            d3.select(comparisonSvgRef.current).selectAll('*').remove();

            const svg = d3.select(comparisonSvgRef.current)
                .attr('width', width)
                .attr('height', height);

            svg.append('g')
                .attr('transform', `translate(0,${height - margin.bottom})`)
                .call(d3.axisBottom(x0))
                .selectAll("text")
                .attr("y", 10)
                .style("text-anchor", "middle");

            svg.append("text")
                .attr("text-anchor", "middle")
                .attr("x", width / 2)
                .attr("y", height - margin.bottom + 50)
                .attr("fill", "black")
                .text("Suburb");

            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(yScale));

            svg.append("text")
                .attr("text-anchor", "middle")
                .attr("transform", `rotate(-90)`)
                .attr("x", -height / 2)
                .attr("y", margin.left - 40)
                .attr("fill", "black")
                .text("Price per week (AUD)");

            const tooltip = d3.select(".prediction-container")
                .append("div")
                .attr("class", "tooltip")
                .style("opacity", 0)
                .style("position", "absolute")
                .style("background-color", "#fff")
                .style("border", "1px solid #d3d3d3")
                .style("padding", "5px")
                .style("border-radius", "5px")
                .style("pointer-events", "none");

            svg.append("g")
                .selectAll("g")
                .data(groupedData)
                .join("g")
                .attr("transform", d => `translate(${x0(d[0])},0)`)
                .selectAll("rect")
                .data(d => d[1])
                .join("rect")
                .attr("x", d => x1(d.housing_type))
                .attr("y", d => yScale(d.price))
                .attr("width", x1.bandwidth())
                .attr("height", d => height - margin.bottom - yScale(d.price))
                .attr("fill", d => colorScale(d.housing_type))
                .on("mouseover", (event, d) => {
                    tooltip.transition().duration(200).style("opacity", 0.9);
                    tooltip.html(`Type: ${d.housing_type}<br>Price: AUD ${d.price.toFixed(2)}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 20) + "px");
                })
                .on("mouseout", () => {
                    tooltip.transition().duration(500).style("opacity", 0);
                })
                .on("mousemove", (event) => {
                    tooltip.style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 20) + "px");
                });

            const legend = svg.append("g")
                .attr("transform", `translate(${width - margin.right + 30}, ${margin.top})`);

            legend.selectAll("rect")
                .data(["Flat", "House"])
                .join("rect")
                .attr("x", 0)
                .attr("y", (d, i) => i * 20)
                .attr("width", 15)
                .attr("height", 15)
                .attr("fill", d => colorScale(d));

            legend.selectAll("text")
                .data(["Flat", "House"])
                .join("text")
                .attr("x", 20)
                .attr("y", (d, i) => i * 20 + 12)
                .text(d => d);
        }
    }, [rentComparisons]);


    // Conditional rendering based on loading state
    return (
        <div className='prediction-container'>
        {loading ? (
            <div className="loading-container">
                <div className="loading-spinner"></div>
            </div>
            ) : (
                <>
                    <h2>Rental forecast for a {numRooms} bedroom {houseType} in {normalizedSuburb} for the next {rentalPeriod} months</h2>
                    <h3>Average rent per week would be {predictedPrice} AUD</h3>
                    <div className='chart-container'>
                        <div className='chart'>
                            <h4>Recent historical rent prices in the suburb</h4>
                            <svg ref={svgRef}></svg>
                        </div>
                        <div className='chart'>
                            <h4>Rent Comparisons in Nearby Suburbs within 3km radius</h4>
                            <svg ref={comparisonSvgRef}></svg>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default GeneratedRentPrediction;
