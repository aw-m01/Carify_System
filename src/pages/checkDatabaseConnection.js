const { Pool } = require('pg');

// PostgreSQL connection configuration
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'Carify', 
  password: '25gr14', 
  port: 5432,
});

// Function to check the database connection
async function checkDatabaseConnection() {
  try {
    // Test the connection by running a simple query
    const res = await pool.query('SELECT NOW()');
    console.log('Connected to the database successfully!');
    console.log('Current time from the database:', res.rows[0].now);
  } catch (err) {
    console.error('Error connecting to the database:', err);
  } finally {
    pool.end();
  }
}

// Run the function to check the database connection
checkDatabaseConnection();
