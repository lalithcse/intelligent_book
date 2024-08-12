# Intelligent Book Management System

## Description

Intelligent Book Management System is a comprehensive solution designed to streamline book management processes within libraries, schools, and personal collections. It offers features such as inventory tracking, borrower management, reservation systems, and automated reminders for due dates. The system aims to enhance accessibility and efficiency in managing books and related resources.

## Installation

### Prerequisites

- Python 3.x
- Django 3.2+
- PostgreSQL database

### Steps

1. Clone the repository to your local machine.
git clone https://github.com/yourusername/intelligent-book.git cd intelligent-book


2. Create a virtual environment and activate it.
python3 -m venv venv source venv/bin/activate # On Windows use venv\Scripts\activate


3. Install the required packages.
pip install -r requirements.txt


4. Set up the database.
python manage.py migrate


5. Collect static files.
python manage.py collectstatic


6. Run the server.
python manage.py runserver


## Usage

After setting up the server, navigate to `http://127.0.0.1:8000` in your web browser to access the Intelligent Book Management System dashboard. 
From here, you can manage books, borrowers, reservations, and view reports.
Please run ./build.sh to create the docker setup

## Contributing

Contributions to the Intelligent Book Management System are welcome. Please feel free to submit pull requests or report issues through the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For support or feedback, please contact us at support@intelligentbook.com.
