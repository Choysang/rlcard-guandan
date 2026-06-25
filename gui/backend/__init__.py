"""Web GUI backend for guandan-rlcard.

A thin Flask + Socket.IO server that drives the open ``guandan_rlcard``
environment for human-vs-AI play in the browser. The engine itself lives
in the :mod:`guandan_rlcard` package; this backend only orchestrates
rooms, turns and the JSON contract the React frontend consumes.
"""
