-- 003: index the singleton transformation_definer agent so future M>1 runs
-- (multiple definers per task for pass@k experiments) slot in cleanly.

UPDATE definers
SET agent = 'transformation_definer_0'
WHERE agent = 'transformation_definer';
