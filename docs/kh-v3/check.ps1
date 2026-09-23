# Compile-check a .cypher file against a real Neo4j, without running it.
#
#   .\check.ps1 01-direct-permissions.cypher
#   .\check.ps1                      # checks every .cypher in this folder
#
# EXPLAIN parses, semantically validates and plans a query but executes
# nothing and touches no data. It catches syntax errors, unknown functions,
# undefined variables, bad label expressions and type mistakes the planner
# can see.
#
# Two details worth knowing:
#  * EXPLAIN still needs parameter VALUES to exist, so every $param in the
#    file is detected and given a dummy. Names ending in Ids/ids or plural
#    get an empty list, everything else an empty string -- override in
#    $ParamOverrides when that guess is wrong.
#  * The query is written to a file and run with `cypher-shell -f`. Passing
#    Cypher inline through PowerShell mangles the quoting and silently
#    truncates the statement, which looks like a syntax error in your query
#    rather than in the harness.

param(
    [string]$File,
    [string]$Container = 'kh-perm-neo4j-1'   # the throwaway harness instance
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Params whose dummy value the heuristic gets wrong go here.
$ParamOverrides = @{
    'types' = "['PARENT_CHILD', 'ATTACHMENT']"
    'skipChecks' = 'false'
    'allowStrict' = 'true'
    'kh_limit' = '50'
}

function Check-One($path) {
    $name = Split-Path -Leaf $path
    $cypher = Get-Content $path -Raw

    # Keep only the first statement: everything up to the first semicolon that
    # ends a statement. The trailing commentary in these files is not Cypher.
    $stmt = ($cypher -split '(?m);\s*$')[0]

    $names = [regex]::Matches($stmt, '\$([A-Za-z_][A-Za-z0-9_]*)') |
             ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
    $lines = foreach ($n in $names) {
        if ($ParamOverrides.ContainsKey($n)) { $v = $ParamOverrides[$n] }
        elseif ($n -match '([Ii]ds|s)$')     { $v = '[]' }
        else                                  { $v = "''" }
        ":param $n => $v"
    }

    $payload = ($lines -join "`n") + "`n`nEXPLAIN`n" + $stmt + "`n;`n"
    $tmp = Join-Path $env:TEMP 'kh_check.cypher'
    Set-Content -Path $tmp -Value $payload -Encoding ascii

    docker cp $tmp "${Container}:/tmp/kh_check.cypher" 2>&1 | Out-Null
    $out = docker exec $Container sh -lc 'P=${NEO4J_AUTH#*/}; /var/lib/neo4j/bin/cypher-shell -u neo4j -p "$P" --format plain -f /tmp/kh_check.cypher' 2>&1
    $text = ($out | Out-String).Trim()

    if ($LASTEXITCODE -eq 0) {
        "PASS  $name   (params: $($names -join ', '))"
    } else {
        "FAIL  $name"
        ($text -split "`n" | Select-Object -First 12) -join "`n"
        ""
    }
}

if ($File) {
    if (-not (Test-Path $File)) { $File = Join-Path $here $File }
    Check-One $File
} else {
    Get-ChildItem -Path $here -Filter *.cypher | ForEach-Object { Check-One $_.FullName }
}
