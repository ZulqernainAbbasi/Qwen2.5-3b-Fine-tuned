from backend import db


def test_booking_roundtrip(tmp_path):
    p = tmp_path / "b.db"
    db.init_db(p)
    bid = db.add_booking("test_drive", path=p, name="Ali", phone="03001234567",
                         model="Toyota Corolla", city="Lahore", date="2030-01-05", km=None)
    assert bid == 1
    rows = db.list_bookings(p)
    assert rows[0]["kind"] == "test_drive" and rows[0]["name"] == "Ali"


def test_ticket_roundtrip(tmp_path):
    p = tmp_path / "b.db"
    db.init_db(p)
    tid = db.add_ticket("Customer angry about delivery delay", path=p)
    assert tid == 1 and db.list_tickets(p)[0]["summary"].startswith("Customer")
