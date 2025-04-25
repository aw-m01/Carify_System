import React, { useEffect, useState } from "react";
import { SearchIcon, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import axios from "axios"; // Add axios import
import "./Reports.css";

const Reports = () => {
    const [reports, setReports] = useState([]); // All reports fetched from the API
    const [searchQuery, setSearchQuery] = useState(""); // Input field value
    const [filteredReports, setFilteredReports] = useState([]); // Filtered reports to display
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchReports = async () => {
            try {
                const response = await axios.get("http://localhost:8000/api/reports", {
                    withCredentials: true, // Use session cookies for authentication
                });
                setReports(response.data);
                setFilteredReports(response.data); // Initially display all reports
                setError("");
            } catch (err) {
                console.error("Error fetching reports: ", err);
                setError(err.message);
                if (err.response?.status === 401) {
                    window.location = "/login"; // Redirect if unauthorized
                }
            } finally {
                setLoading(false);
            }
        };

        fetchReports();
    }, []); // Empty dependency array since we fetch once on mount

    // Function to handle deleting a report
    const handleDeleteReport = async (plateNumber) => {
        if (!window.confirm("Are you sure you want to delete this report?")) {
            return;
        }
        try {
          await axios.delete(`http://localhost:8000/api/cars/${plateNumber}`, {
              withCredentials: true,
          });

          // Remove the deleted car's reports from both states
          setReports((prevReports) =>
              prevReports.filter((report) => report.Plate_Number !== plateNumber)
          );
          setFilteredReports((prevFilteredReports) =>
              prevFilteredReports.filter((report) => report.Plate_Number !== plateNumber)
          );

          alert("Car and associated reports deleted successfully.");
      } catch (err) {
          console.error("Error deleting car: ", err);
          alert("Failed to delete the car. Please try again.");
      }
       
    };

    // Function to handle the search button click
    const handleSearch = () => {
        const filtered = reports.filter(
            (report) =>
                report.Report_ID?.toString().includes(searchQuery) ||
                report.Plate_Number?.toLowerCase().includes(searchQuery.toLowerCase())
        );
        setFilteredReports(filtered); // Update the filtered reports state
    };
    // Function to handle key press in the input field
    const handleKeyPress = (e) => {
      if (e.key === "Enter") {
          handleSearch(); // Trigger search when Enter key is pressed
      }
  };

    return (
        <div>
            <Navbar />
            <div className="reports-container">
                <div className="search-container">
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Enter Report ID or Plate Number"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={handleKeyPress} // Listen for key presses

                    />
                    <button className="search-button" onClick={handleSearch}>
                        <SearchIcon size={20} />
                    </button>
                </div>

                <Link to="/new-report">
                    <button className="new-report-button">New Report</button>
                </Link>
            </div>

            {/* Display Reports */}
            {loading ? (
                <div className="loading-message">Loading reports...</div>
            ) : error ? (
                <div className="error-message">{error}</div>
            ) : (
                <div className="reports-grid">
                    {filteredReports.map((report) => (
                        <div key={report.Report_ID} className="report-card">
                            <h3>Report ID: {report.Report_ID}</h3>
                            <hr />
                            <p>
                                <strong>Status:</strong>{" "}
                                <span
                                    className={
                                        report.Found ? "status-found" : "status-not-found"
                                    }
                                >
                                    {report.Found ? "Found" : "Not Found"}
                                </span>
                            </p>
                            <p><strong>Plate Number:</strong> {report.Plate_Number}</p>
                            <p><strong>Model:</strong> {report.Model}</p>
                            <p><strong>Color:</strong> {report.Color}</p>
                            <p><strong>Made by:</strong> {report.OfficerName}</p>

                            <div className="button-group">
                                <Link to={`/report/${report.Report_ID}`}>
                                    <button className="view-records-button">View Records</button>
                                </Link>
                                <button
                                    className="delete-report-button"
                                    onClick={() => handleDeleteReport(report.Plate_Number)}
                                >
                                    <Trash2 size={16} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Reports;