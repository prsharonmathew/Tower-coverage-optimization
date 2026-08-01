import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = path.resolve("..");
const validationCsvPath = path.join(projectRoot, "output", "selected_tower_restriction_validation_germany.csv");
const outputPath = path.join(projectRoot, "output", "selected_tower_hard_restrictions_germany_relaxed.xlsx");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(value);
      if (row.some((cell) => cell !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value !== "" || row.length > 0) {
    row.push(value);
    rows.push(row);
  }

  const [headers, ...dataRows] = rows;
  return dataRows.map((dataRow) => Object.fromEntries(headers.map((header, index) => [header, dataRow[index] ?? ""])));
}

function asBoolean(value) {
  return String(value).trim().toLowerCase() === "true";
}

function prettyLayerName(layerName) {
  return String(layerName)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function uniqueSorted(values) {
  return [...new Set(values)].sort();
}

function setHeaderStyle(range, fill = "#0F766E") {
  range.format = {
    fill,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#CBD5E1" },
  };
}

function setBodyStyle(range) {
  range.format = {
    borders: { preset: "all", style: "thin", color: "#E2E8F0" },
    verticalAlignment: "top",
    wrapText: true,
  };
}

const csvText = await fs.readFile(validationCsvPath, "utf8");
const records = parseCsv(csvText);
const hardRecords = records.filter((row) => asBoolean(row.is_hard_restriction));

const towerGroups = new Map();
for (const row of hardRecords) {
  if (!towerGroups.has(row.tower_id)) {
    towerGroups.set(row.tower_id, []);
  }
  towerGroups.get(row.tower_id).push(row);
}

const summaryRows = [...towerGroups.entries()]
  .sort((a, b) => Number(a[0]) - Number(b[0]))
  .map(([towerId, rows]) => {
    const first = rows[0];
    const layerNames = uniqueSorted(rows.map((row) => row.layer_name));
    const violationTypes = uniqueSorted(rows.map((row) => row.violation_type));
    return [
      Number(towerId),
      Number(first.latitude),
      Number(first.longitude),
      Number(first.radius),
      layerNames.map(prettyLayerName).join(", "),
      violationTypes.map(prettyLayerName).join(", "),
      rows.length,
      violationTypes.includes("outside_country_boundary"),
      layerNames.includes("water_bodies"),
      layerNames.includes("protected_areas_strict"),
      layerNames.includes("airports"),
      layerNames.includes("aviation_radar_protection"),
      layerNames.includes("military_restricted"),
    ];
  });

const detailRows = hardRecords
  .sort((a, b) => Number(a.tower_id) - Number(b.tower_id) || String(a.layer_name).localeCompare(String(b.layer_name)))
  .map((row) => [
    Number(row.tower_id),
    Number(row.latitude),
    Number(row.longitude),
    Number(row.radius),
    prettyLayerName(row.violation_type),
    prettyLayerName(row.layer_name),
    row.category,
    asBoolean(row.is_hard_restriction),
    asBoolean(row.is_soft_restriction),
  ]);

const countsByLayer = new Map();
for (const row of hardRecords) {
  const layerName = row.layer_name;
  if (!countsByLayer.has(layerName)) {
    countsByLayer.set(layerName, { rows: 0, towerIds: new Set() });
  }
  const entry = countsByLayer.get(layerName);
  entry.rows += 1;
  entry.towerIds.add(row.tower_id);
}

const countRows = [...countsByLayer.entries()]
  .sort((a, b) => a[0].localeCompare(b[0]))
  .map(([layerName, entry]) => [
    prettyLayerName(layerName),
    entry.towerIds.size,
    entry.rows,
    uniqueSorted([...entry.towerIds].map(Number)).join(", "),
  ]);

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Hard Violation Towers");
const detailsSheet = workbook.worksheets.add("Violation Details");
const countsSheet = workbook.worksheets.add("Counts By Restriction");

summarySheet.showGridLines = false;
detailsSheet.showGridLines = false;
countsSheet.showGridLines = false;

summarySheet.getRange("A1:M1").merge();
summarySheet.getRange("A1").values = [["Selected Towers Inside Hard Restricted Areas"]];
summarySheet.getRange("A1").format = {
  fill: "#0F172A",
  font: { bold: true, color: "#FFFFFF", size: 14 },
  horizontalAlignment: "center",
};
summarySheet.getRange("A2:M2").merge();
summarySheet.getRange("A2").values = [[
  `Germany validation result: ${summaryRows.length} selected towers have hard restriction violations. Some towers can appear in more than one restriction type.`,
]];
summarySheet.getRange("A2").format = {
  fill: "#E0F2FE",
  font: { color: "#0F172A" },
  wrapText: true,
};

const summaryHeaders = [
  "tower_id",
  "latitude",
  "longitude",
  "radius_km",
  "restricted_area_types",
  "violation_types",
  "num_violations",
  "outside_country_boundary",
  "in_water_bodies",
  "in_protected_areas_strict",
  "in_airports",
  "in_aviation_radar",
  "in_military_restricted",
];
summarySheet.getRange("A4:M4").values = [summaryHeaders];
summarySheet.getRangeByIndexes(4, 0, summaryRows.length, summaryHeaders.length).values = summaryRows;
setHeaderStyle(summarySheet.getRange("A4:M4"));
setBodyStyle(summarySheet.getRangeByIndexes(4, 0, summaryRows.length, summaryHeaders.length));
summarySheet.getRange("B5:C200").format.numberFormat = "0.000000";
summarySheet.getRange("D5:D200").format.numberFormat = "0.0";
summarySheet.getRange("G5:G200").format.numberFormat = "0";
summarySheet.freezePanes.freezeRows(4);
summarySheet.getRange("A:M").format.autofitColumns();

detailsSheet.getRange("A1:I1").merge();
detailsSheet.getRange("A1").values = [["Detailed Hard Restriction Violations"]];
detailsSheet.getRange("A1").format = {
  fill: "#7F1D1D",
  font: { bold: true, color: "#FFFFFF", size: 14 },
  horizontalAlignment: "center",
};
const detailHeaders = [
  "tower_id",
  "latitude",
  "longitude",
  "radius_km",
  "violation_type",
  "layer_name",
  "category",
  "is_hard_restriction",
  "is_soft_restriction",
];
detailsSheet.getRange("A3:I3").values = [detailHeaders];
detailsSheet.getRangeByIndexes(3, 0, detailRows.length, detailHeaders.length).values = detailRows;
setHeaderStyle(detailsSheet.getRange("A3:I3"), "#991B1B");
setBodyStyle(detailsSheet.getRangeByIndexes(3, 0, detailRows.length, detailHeaders.length));
detailsSheet.getRange("B4:C200").format.numberFormat = "0.000000";
detailsSheet.getRange("D4:D200").format.numberFormat = "0.0";
detailsSheet.freezePanes.freezeRows(3);
detailsSheet.getRange("A:I").format.autofitColumns();

countsSheet.getRange("A1:D1").merge();
countsSheet.getRange("A1").values = [["Restriction Type Counts"]];
countsSheet.getRange("A1").format = {
  fill: "#334155",
  font: { bold: true, color: "#FFFFFF", size: 14 },
  horizontalAlignment: "center",
};
const countHeaders = ["restricted_area_type", "unique_towers", "violation_rows", "tower_ids"];
countsSheet.getRange("A3:D3").values = [countHeaders];
countsSheet.getRangeByIndexes(3, 0, countRows.length, countHeaders.length).values = countRows;
setHeaderStyle(countsSheet.getRange("A3:D3"), "#334155");
setBodyStyle(countsSheet.getRangeByIndexes(3, 0, countRows.length, countHeaders.length));
countsSheet.getRange("B4:C50").format.numberFormat = "0";
countsSheet.freezePanes.freezeRows(3);
countsSheet.getRange("A:D").format.autofitColumns();

await workbook.render({ sheetName: "Hard Violation Towers", autoCrop: "all", scale: 1, format: "png" });
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
});
if (errorScan.ndjson.includes("#")) {
  console.log(errorScan.ndjson);
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
