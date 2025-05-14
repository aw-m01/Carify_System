import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import Navbar from "../components/Navbar";
import "./ReportRecords.css";

const ReportRecords = () => {
  const { id } = useParams();
  const [records, setRecords] = useState([]);
  const [reportDetails, setReportDetails] = useState(null); // New state for report details
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedRecord, setSelectedRecord] = useState(null);

  

  useEffect(() => {
    const fetchReportDetails = async () => {
      try {
        console.log(`Fetching report details for Report ID: ${id}`);
        const response = await fetch(`http://localhost:8000/api/reports`, {
          credentials: "include",
        });

        if (!response.ok) {
          const errorData = await response.json();
          console.error(
            `Failed to fetch report details: ${response.status} - ${errorData.detail}`
          );
          throw new Error(errorData.detail || "Failed to fetch report details");
        }

        const data = await response.json();
        const report = data.find((r) => r.Report_ID === parseInt(id));
        if (!report) {
          throw new Error("Report not found");
        }
        setReportDetails(report);
      } catch (error) {
        console.error("Error fetching report details:", error.message);
        setError(error.message);
      }
    };
    const fetchRecords = async () => {
      try {
        console.log(`Fetching records for Report ID: ${id}`);
        const response = await fetch(`http://localhost:8000/api/records/${id}`, {
          credentials: "include",
        });

        if (!response.ok) {
          const errorData = await response.json();
          console.error(
            `Failed to fetch records: ${response.status} - ${errorData.detail}`
          );
          throw new Error(errorData.detail || "Failed to fetch records");
        }

        const data = await response.json();
        console.log("Records fetched successfully:", data);
        // Sort records by Record_ID in descending order (highest to lowest)
        const sortedData = data.sort((a, b) => b.Record_ID - a.Record_ID);
        setRecords(sortedData);
      } catch (error) {
        console.error("Error fetching records:", error.message);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };
    fetchReportDetails();
    fetchRecords();
  }, [id]);

  useEffect(() => {
    if (records.length > 0 && !selectedRecord) {
      setSelectedRecord(records[0]); // Select the record with the highest Record_ID
    }
  }, [records, selectedRecord]);

  const handleRecordClick = (record) => {
    setSelectedRecord(record);
  };

  const getRecordIndex = (record) => {
    const index = records.indexOf(record); // 0-based index in the sorted array
    return records.length - index; // Convert to descending order 
  };

  return (
    <div>
      <Navbar />
      <div className="report-records-container">
          {/* Report Details Section */}
          {reportDetails ? (
          <div className="report-details-box">
            <h3>Report #{reportDetails.Report_ID}</h3>
            <p>
              <strong>Plate Number:</strong> {reportDetails.Plate_Number}
            </p>
           
            <p>
              <strong>Model:</strong> {reportDetails.Model}
            </p>
           
            <p>
              <strong>Color:</strong> {reportDetails.Color}
            </p>
            <p>
              <strong>Status:</strong>{" "}
              <span
                className={
                  reportDetails.Found ? "status-found" : "status-not-found"
                }
              >
                {reportDetails.Found ? "Found" : "Not Found"}
              </span>
            </p>
          </div>
        ) : (
          <div className="loading-indicator">Loading report details...</div>
        )}

        <h2 className="report-title">Report #{id} - Detection Records</h2>

        {loading ? (
          <div className="loading-indicator">Loading records...</div>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : records.length > 0 ? (
          <div className="container">
            {/* Left Panel: Detailed View of Selected Record */}
            <div className="left-panel">
              {selectedRecord ? (
                <div className="record-details">
                  <h3>Record {getRecordIndex(selectedRecord)} of Report #{id}</h3>
                  <p className="detail-item">
                    <span className="detail-label">Detection Time:</span>
                    <span className="detail-value timestamp">
                      {selectedRecord.Detection_Time}
                    </span>
                  </p>
                  <p className="detail-item">
                    <span className="detail-label">Detection Location:</span>
                    <span className="detail-value location">
                    <a
                        href={selectedRecord.Detection_Location}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                      {selectedRecord.Detection_Location}
                      </a>
                    </span>
                  </p>
                  {/* Display Car Image */}
                  {selectedRecord.Car_Image && (
                    <div className="detail-item">
                      <span className="detail-label">Car Image:</span>
                      <img
                        src={selectedRecord.Car_Image}
                        alt="Detected Car"
                        className="car-image"
                        onError={(e) =>
                          (e.target.src =
                            "http://localhost:8000/detections/placeholder.jpg")
                        }
                      />
                    </div>
                  )}
                </div>
              ) : (
                <p>Select a record from the list to view details.</p>
              )}
            </div>

            {/* Right Panel: List of Records (Highest Record_ID at the Top) */}
            <div className="right-panel">
              {records.map((record) => (
                <div
                  key={record.Record_ID}
                  className={`record-card ${
                    selectedRecord && selectedRecord.Record_ID === record.Record_ID
                      ? "selected"
                      : ""
                  }`}
                  onClick={() => handleRecordClick(record)}
                >
                  <p>
                    <strong>Record:</strong> {getRecordIndex(record)}
                  </p>
                  <p>
                    <strong>Detection Time:</strong>{" "}
                    {record.Detection_Time}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="no-records">No detection records found</div>
        )}
      </div>
    </div>
  );
};

export default ReportRecords;
