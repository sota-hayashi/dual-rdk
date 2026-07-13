<?php
/**
 * Minimal upload endpoint for jsPsych data.
 *
 * Expects JSON payload:
 *   { "subject_id": "subj01", "data": "...CSV or JSON string..." }
 *
 * Saves to /com05/com05hay/data/<subject_id>.json
 * Adjust $saveDir if必要.
 */

// upload.php（デバッグ用）
// ini_set('display_errors', '1');
// ini_set('log_errors', '1');
// ini_set('error_log', '/com05/com05hay/php_error.log');
// error_reporting(E_ALL);


// ---- config ----
$saveDir = "/com05/com05hay/data_online_experiment";

// ---- read request body ----
$raw = file_get_contents("php://input");
if ($raw === false || $raw === "") {
    http_response_code(400);
    exit("no data");
}

$payload = json_decode($raw, true);
if (!is_array($payload) || !isset($payload["subject_id"], $payload["data"])) {
    http_response_code(400);
    exit("invalid payload");
}

// ---- build filename ----
$subject_id = $payload["subject_id"];
$filename = "{$subject_id}.json";
$path = rtrim($saveDir, "/") . "/" . $filename;

// ---- ensure dir is writable ----
if (!is_dir($saveDir) || !is_writable($saveDir)) {
    http_response_code(500);
    exit("save dir not writable");
}

// ---- write file ----
$bytes = file_put_contents($path, $payload["data"]);
if ($bytes === false) {
    http_response_code(500);
    exit("write failed");
}

http_response_code(200);
echo "ok";

// error_log("hit upload.php");
// error_log("saveDir exists=".(int)is_dir($saveDir)." writable=".(int)is_writable($saveDir));

