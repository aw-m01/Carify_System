from fastapi import FastAPI, HTTPException, Depends, Request, status, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware  # Corrected import
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from asyncpg import Connection
import asyncpg
import os
from dotenv import load_dotenv
import logging
from passlib.context import CryptContext



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Vehicle Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware
SECRET_KEY = os.getenv("SESSION_SECRET", "your-secret-key")
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session_id",
    max_age=3600,  # 1 hour
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class DatabaseConfig:
    def __init__(self):
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.database = os.getenv('DB_NAME')
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('DB_PORT')
        self.pool = None

    async def get_pool(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                min_size=1,
                max_size=10,
                user=self.user,
                password=self.password,
                database=self.database,
                host=self.host,
                port=self.port
            )
        return self.pool

db_config = DatabaseConfig()

# Pydantic models
class UserSignup(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class DetectionData(BaseModel):
    plate_number: str
    color: str
    model: str  # Add this field
    timestamp: str
    location: str
    Car_Image: str  # Full URL of the image
    plate_image: str  # Full URL of the image





class ReportData(BaseModel):
    plateNumber: str
    model: str
    color: str






# Dependency to get database connection
async def get_db():
    pool = await db_config.get_pool()
    async with pool.acquire() as conn:
        yield conn

# Helper function to authenticate users
async def authenticate_user(email: str, password: str, conn):
    user = await conn.fetchrow(
        'SELECT * FROM "Traffic Police" WHERE "Email" = $1', email
    )
    if not user or not pwd_context.verify(password, user["Password"]):
        return None
    return user


app.mount("/static-detections", StaticFiles(directory="detections"), name="static-detections")

@app.post("/api/signup")
async def signup(user: UserSignup, db: asyncpg.Connection = Depends(get_db)):
    # Check if user already exists
    existing_user = await db.fetchrow(
        'SELECT * FROM "Traffic Police" WHERE "Email" = $1', user.email
    )
    if existing_user:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"message": "This email is already registered. Please log in."}
        )
    # Hash password and create user
    hashed_password = pwd_context.hash(user.password)
    await db.execute(
        'INSERT INTO "Traffic Police" ("Name", "Email", "Password") VALUES ($1, $2, $3)',
        user.name, user.email, hashed_password
    )
    return JSONResponse(content={"message": "User registered successfully"})

@app.post("/api/login")
async def login(user: UserLogin, request: Request, db: asyncpg.Connection = Depends(get_db)):
    # Authenticate user
    authenticated_user = await authenticate_user(user.email, user.password, db)
    if not authenticated_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    # Store user info in session
    request.session["user"] = {
        "user_id": authenticated_user["Police_ID"],
        "name": authenticated_user["Name"],
        "email": authenticated_user["Email"],
    }
    return JSONResponse(content={"message": "Login successful"})

@app.post("/api/logout")
async def logout(request: Request):
    if "user" in request.session:
        del request.session["user"]
    return JSONResponse(content={"message": "Logout successful"})



@app.get("/api/me")
async def me(request: Request):
    if "user" not in request.session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return JSONResponse(content={
        "name": request.session["user"]["name"],
        "email": request.session["user"]["email"]
    })

# Assuming get_db is already defined in main.py
@app.get("/api/reports")
async def get_reports(request: Request, db: Connection = Depends(get_db)):
    # Check if the user is authenticated via session
    if "user" not in request.session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Execute the database query
    try:
        query = '''
        SELECT 
            cr."Report_ID", 
            cr."Plate_Number", 
            c."Model", 
            c."Color", 
            cr."Found", 
            tp."Name" as "OfficerName"
        FROM "Car Report" cr 
        JOIN "Car" c ON cr."Plate_Number" = c."Plate_Number"
        JOIN "Traffic Police" tp ON cr."Police_ID" = tp."Police_ID"
        '''
        reports = await db.fetch(query)
        return reports  # FastAPI will serialize this to JSON automatically
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching reports")


@app.get("/api/records/{reportId}")
async def get_records(reportId: int, request: Request, db: Connection = Depends(get_db)):
    logger.info(f"Fetching records for Report ID: {reportId}")

    # Check if the user is authenticated via session
    if "user" not in request.session:
        logger.warning("User not authenticated")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        query = '''
        SELECT "Record_ID", "Detection_Time", "Detection_Location", "Car_Image"
        FROM "Car Record"
        WHERE "Report_ID" = $1
        '''
        logger.info(f"Executing query: {query} with Report_ID={reportId}")
        records = await db.fetch(query, reportId)
        logger.info(f"Query executed successfully, records: {records}")
        return records or []  # Return empty array if no records found
    except Exception as e:
        logger.error(f"Error fetching records: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching records")


@app.post("/api/report", response_model=dict)
async def create_report(
    request: Request,
    data: ReportData,
    db: Connection = Depends(get_db)
):
    try:
        # Check if the user is authenticated via session
        if "user" not in request.session:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        
        # Extract user data from the session
        police_id = request.session["user"]["user_id"]

        # Extract data from the request body
        plate_number = data.plateNumber
        model = data.model
        color = data.color

        # Validate required fields
        if not all([plate_number, model, color]):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing required fields")

        # Check if a report with the same Plate_Number already exists
        existing_report_query = '''
        SELECT "Report_ID" FROM public."Car Report" WHERE "Plate_Number" = $1
        '''
        existing_report = await db.fetch(existing_report_query, plate_number)

        if existing_report:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A report for this car already exists")

        # Insert car details into the "Car" table (if not already present)
        insert_car_query = '''
        INSERT INTO public."Car" ("Plate_Number", "Model", "Color") 
        VALUES ($1, $2, $3) ON CONFLICT ("Plate_Number") DO NOTHING
        '''
        await db.execute(insert_car_query, plate_number, model, color)

        # Insert the report into the "Car Report" table
        insert_report_query = '''
        INSERT INTO public."Car Report" ("Plate_Number", "Police_ID") 
        VALUES ($1, $2) RETURNING "Report_ID"
        '''
        report = await db.fetchrow(insert_report_query, plate_number, police_id)

        # Return success response with the new report ID
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Report submitted successfully!",
                "reportId": report["Report_ID"]
            }
        )

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        logger.error(f"Error submitting report: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error submitting report")


@app.delete("/api/reports/{reportId}")
async def delete_report(
    reportId: int,
    request: Request,
    db: Connection = Depends(get_db)
):
    """
    Delete a report form the "Car Report" table
    """

    logger.info(f"Attempting to delete report with Report_ID: {reportId}")

    # Check if user is authenticated via session
    if "user" not in request.session:
        logger.warning("User is not authenticated")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        #Check if the report exists
        check_query = '''
        SELECT "Report_ID" FROM "Car Report" Where "Report_ID" = $1        
        '''

        existing_report = await db.fetchrow(check_query, reportId)

        if not existing_report:
            logger.warning(f"Report with ID {reportId} not found")
            raise HTTPException(status_code=404, detail="Report is not found")
        

        # Delete the report
        delete_query = '''
        DELETE FROM "Car Report" WHERE "Report_ID" = $1
        '''

        await db.execute(delete_query, reportId)

        logger.info(f"Report with ID {reportId} deleted successfully")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Report with ID {reportId} deleted successfully"}
        )
    
    except HTTPException as http_err:
        raise http_err
    
    except Exception as e:
        logger.error(f"Error in deleting report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in deleting report")



@app.delete("/api/cars/{plateNumber}")
async def delete_car(
    plateNumber: str,
    request: Request,
    db: Connection = Depends(get_db)
):
    """
    Delete a car from the "Car" table and cascade the deletion to the "Car Report" table.
    """
    logger.info(f"Attempting to delete car with Plate Number: {plateNumber}")
    
    # Check if user is authenticated via session
    if "user" not in request.session:
        logger.warning("User is not authenticated")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Check if the car exists
        check_query = '''
        SELECT "Plate_Number" FROM "Car" WHERE "Plate_Number" = $1
        '''
        car = await db.fetchrow(check_query, plateNumber)
        if not car:
            logger.warning(f"Car with Plate Number {plateNumber} not found")
            raise HTTPException(status_code=404, detail="Car not found")
        
        # Delete the car from the "Car" table
        delete_query = '''
        DELETE FROM "Car" WHERE "Plate_Number" = $1
        '''
        await db.execute(delete_query, plateNumber)
        
        logger.info(f"Car with Plate Number {plateNumber} deleted successfully")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Car with Plate Number {plateNumber} deleted successfully"}
        )
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Error in deleting car: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in deleting car")





# In-memory list of active WebSocket connections
active_connections = []

@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    """
    WebSocket endpoint for real-time notifications.
    """
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)



@app.post("/detections", status_code=201)
async def handle_detection(data: DetectionData, db: Connection = Depends(get_db)):
    """
    Handle detection data and insert records into the "Car Record" table based on the following logic:
    1. If the plate number matches, create a record regardless of car model and color.
    2. If the plate number does not match, check if both car model and color match, and if so, create a record.
    """
    logger.info(f"Received detection data: {data}")
    try:
        # Step 1: Check if the plate number matches any car in the "Car" table
        car = await db.fetchrow(
            '''SELECT "Plate_Number", "Model", "Color" 
               FROM "Car" WHERE "Plate_Number" = $1''',
            data.plate_number
        )
        if car:
            # Plate number matched
            logger.info(f"Plate number matched: {data.plate_number}")
            
            # Confirmation step for car model and color (optional)
            if car["Model"].lower() == data.model.lower() and car["Color"].lower() == data.color.lower():
                logger.info("Car model and color confirmed for Plate Number: {car['Plate_Number']}")
          
            reports = await db.fetch(
                '''SELECT "Report_ID" FROM "Car Report" WHERE "Plate_Number" = $1''',
                car["Plate_Number"]
            )
            if not reports:
                logger.info(f"No reports found for plate: {car['Plate_Number']}")
                return {"message": "No matching reports found for the matched plate number"}
            created_records = 0
            for report in reports:
                record = await db.fetchrow('''
                    INSERT INTO "Car Record" 
                    ("Detection_Time", "Detection_Location", "Report_ID", "Car_Image", "Plate_Image")
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING "Record_ID", "Detection_Time", "Detection_Location", "Report_ID"
                ''', data.timestamp, data.location, report["Report_ID"], data.Car_Image, data.plate_image)
                created_records += 1
                notification_message = {
                    "message": (
                        f"New detection recorded for a car with Plate Number: {car['Plate_Number']}, "
                        f"Report ID: {report['Report_ID']}, Detection Time: {data.timestamp}"
                    ),
                    "timestamp": data.timestamp,
                    "location": data.location,
                    "plate_number": car["Plate_Number"],
                    "report_id": report["Report_ID"],
                    "record_id": record["Record_ID"],
                    "Car_Image": data.Car_Image
                }
                for connection in active_connections:
                    await connection.send_json(notification_message)
            logger.info(f"Created {created_records} new car records for plate number: {data.plate_number}")
            return {"message": f"Created {created_records} records for plate number {data.plate_number}"}
        else:
            # Step 2: Plate number did not match, check car model and color
            logger.info(f"Plate number did not match: {data.plate_number}")
            cars = await db.fetch(
                '''SELECT "Plate_Number", "Model", "Color" 
                   FROM "Car" WHERE LOWER("Color") = LOWER($1)''',
                data.color.lower()
            )
            if not cars:
                logger.warning(f"No cars found with color: {data.color}")
                return {"message": "No matching cars found for the given color"}
            created_records = 0
            for car in cars:
                  if car["Model"].lower() == data.model.lower():
                    logger.info(f"Car model and color matched for Plate Number: {car['Plate_Number']}")
                    
                   
                    reports = await db.fetch(
                        '''SELECT "Report_ID" FROM "Car Report" WHERE "Plate_Number" = $1''',
                        car["Plate_Number"]
                    )
                    if not reports:
                        logger.info(f"No reports found for plate: {car['Plate_Number']}")
                        continue
                    for report in reports:
                        record = await db.fetchrow('''
                            INSERT INTO "Car Record" 
                            ("Detection_Time", "Detection_Location", "Report_ID", "Car_Image", "Plate_Image")
                            VALUES ($1, $2, $3, $4, $5)
                            RETURNING "Record_ID", "Detection_Time", "Detection_Location", "Report_ID"
                        ''', data.timestamp, data.location, report["Report_ID"], data.Car_Image, data.plate_image)
                        created_records += 1
                        notification_message = {
                            "message": (
                                f"New detection recorded for a car with Plate Number: {car['Plate_Number']}, "
                                f"Report ID: {report['Report_ID']}, Detection Time: {data.timestamp}"
                            ),
                            "timestamp": data.timestamp,
                            "location": data.location,
                            "plate_number": car["Plate_Number"],
                            "report_id": report["Report_ID"],
                            "record_id": record["Record_ID"],
                            "Car_Image": data.Car_Image
                        }
                        for connection in active_connections:
                            await connection.send_json(notification_message)
            logger.info(f"Created {created_records} new car records based on car model and color")
            return {"message": f"Created {created_records} records based on car model and color"}
    except asyncpg.PostgresError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")