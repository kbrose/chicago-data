The data is an Nx4 matrix.

Column 1: Route number
The route number is stored as a BASE 36 (!!!!) number to account for the few and unfortunate cases where a letter has squeaked its way into the route number (i.e. X28, J14, ...).

Column 2: The date
This is stored in Matlab serial timestamp.

Column 3: The day types
day types == 1 => 'A', i.e. Saturday
day types == 2 => 'U', i.e. Sunday or Holiday
day types == 3 => 'W', i.e. Weekday

This was determined by the CTA and not by me.

Column 4: Number of rides
Integer.
