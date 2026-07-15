-- Đếm tổng số cột thực tế đang có trong database _dat_dev
SELECT COUNT(*) FROM i26s02004_dat_dev.sys.tables t 
JOIN i26s02004_dat_dev.sys.columns c ON t.object_id = c.object_id 
WHERE t.is_ms_shipped = 0;

-- Đếm tổng số cột thực tế đang có trong database _iot_dev
SELECT COUNT(*) FROM i26s02004_iot_dev.sys.tables t 
JOIN i26s02004_iot_dev.sys.columns c ON t.object_id = c.object_id 
WHERE t.is_ms_shipped = 0;

IF OBJECT_ID('tempdb..#FinalReport') IS NOT NULL
    DROP TABLE #FinalReport;

CREATE TABLE #FinalReport
(
    DatabaseName NVARCHAR(128),
    SchemaName NVARCHAR(128),
    TableName NVARCHAR(128),
    ColumnName NVARCHAR(128),
    DataType NVARCHAR(128),
    RowCounts BIGINT NULL,
    NonNullCount BIGINT NULL,
    MinValue NVARCHAR(MAX) NULL,
    MaxValue NVARCHAR(MAX) NULL,
    SampleValues NVARCHAR(MAX) NULL
);

-- 1. Thu thập Metadata từ các database bằng Dynamic SQL để tránh sai lệch SCHEMA_NAME
DECLARE @MetaSql NVARCHAR(MAX) = N'
INSERT INTO #FinalReport (DatabaseName, SchemaName, TableName, ColumnName, DataType)
SELECT 
    ''i26s02004_dat_dev'',
    s.name,
    t.name,
    c.name,
    ty.name
FROM i26s02004_dat_dev.sys.tables t
JOIN i26s02004_dat_dev.sys.schemas s ON t.schema_id = s.schema_id
JOIN i26s02004_dat_dev.sys.columns c ON t.object_id = c.object_id
JOIN i26s02004_dat_dev.sys.types ty ON c.user_type_id = ty.user_type_id
WHERE t.is_ms_shipped = 0;

INSERT INTO #FinalReport (DatabaseName, SchemaName, TableName, ColumnName, DataType)
SELECT 
    ''i26s02004_iot_dev'',
    s.name,
    t.name,
    c.name,
    ty.name
FROM i26s02004_iot_dev.sys.tables t
JOIN i26s02004_iot_dev.sys.schemas s ON t.schema_id = s.schema_id
JOIN i26s02004_iot_dev.sys.columns c ON t.object_id = c.object_id
JOIN i26s02004_iot_dev.sys.types ty ON c.user_type_id = ty.user_type_id
WHERE t.is_ms_shipped = 0;
';

EXEC sp_executesql @MetaSql;

-- 2. Duyệt Cursor và tối ưu hóa việc quét dữ liệu
DECLARE
    @DbName NVARCHAR(128),
    @SchemaName NVARCHAR(128),
    @TableName NVARCHAR(128),
    @ColumnName NVARCHAR(128),
    @DataType NVARCHAR(128),
    @Sql NVARCHAR(MAX);

DECLARE cur CURSOR LOCAL FAST_FORWARD FOR
SELECT DatabaseName, SchemaName, TableName, ColumnName, DataType
FROM #FinalReport;

OPEN cur;

FETCH NEXT FROM cur INTO @DbName, @SchemaName, @TableName, @ColumnName, @DataType;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Nhóm các kiểu dữ liệu KHÔNG THỂ lấy Min/Max hoặc không nên lấy Sample trực tiếp
    IF @DataType IN ('image', 'text', 'ntext', 'xml', 'timestamp', 'geometry', 'geography', 'hierarchyid', 'binary', 'varbinary', 'sql_variant')
    BEGIN
        SET @Sql = N'
        UPDATE #FinalReport
        SET 
            RowCounts = (SELECT COUNT(*) FROM [' + @DbName + '].[' + @SchemaName + '].[' + @TableName + ']),
            NonNullCount = (SELECT COUNT([' + @ColumnName + ']) FROM [' + @DbName + '].[' + @SchemaName + '].[' + @TableName + ']),
            MinValue = N''[Không hỗ trợ kdl này]'',
            MaxValue = N''[Không hỗ trợ kdl này]'',
            SampleValues = N''[Không hỗ trợ kdl này]''
        WHERE DatabaseName = ''' + @DbName + ''' AND SchemaName = ''' + @SchemaName + ''' 
          AND TableName = ''' + @TableName + ''' AND ColumnName = ''' + @ColumnName + ''';';
    END
    ELSE
    BEGIN
        SET @Sql = N'
        DECLARE @Rows BIGINT, @NonNull BIGINT, @MinVal NVARCHAR(MAX), @MaxVal NVARCHAR(MAX), @Samples NVARCHAR(MAX);

        SELECT 
            @Rows = COUNT(*),
            @NonNull = COUNT([' + @ColumnName + ']),
            @MinVal = CAST(MIN([' + @ColumnName + ']) AS NVARCHAR(MAX)),
            @MaxVal = CAST(MAX([' + @ColumnName + ']) AS NVARCHAR(MAX))
        FROM [' + @DbName + '].[' + @SchemaName + '].[' + @TableName + '];

        SELECT @Samples = STRING_AGG(CAST(Val AS NVARCHAR(MAX)), '', '')
        FROM (
            SELECT DISTINCT TOP 3 [' + @ColumnName + '] AS Val
            FROM [' + @DbName + '].[' + @SchemaName + '].[' + @TableName + ']
            WHERE [' + @ColumnName + '] IS NOT NULL
        ) X;

        UPDATE #FinalReport
        SET 
            RowCounts = @Rows,
            NonNullCount = @NonNull,
            MinValue = @MinVal,
            MaxValue = @MaxVal,
            SampleValues = @Samples
        WHERE DatabaseName = ''' + @DbName + ''' AND SchemaName = ''' + @SchemaName + ''' 
          AND TableName = ''' + @TableName + ''' AND ColumnName = ''' + @ColumnName + ''';';
    END

    BEGIN TRY
        EXEC sp_executesql @Sql;
    END TRY
    BEGIN CATCH
        UPDATE #FinalReport
        SET 
            SampleValues = N'Lỗi đọc dữ liệu: ' + LEFT(ERROR_MESSAGE(), 100)
        WHERE DatabaseName = @DbName AND SchemaName = @SchemaName 
          AND TableName = @TableName AND ColumnName = @ColumnName;
    END CATCH;

    FETCH NEXT FROM cur INTO @DbName, @SchemaName, @TableName, @ColumnName, @DataType;
END

CLOSE cur;
DEALLOCATE cur;

-- =====================================================
-- FINAL REPORT (Đã sửa lỗi cú pháp dấu phẩy và sai tên biến)
-- =====================================================
SELECT
    DatabaseName AS [Database],
    SchemaName AS [Schema],
    TableName AS [Table],
    ColumnName AS [Column],
    DataType AS [DataType],
    RowCounts,
    NonNullCount,
    CASE
        WHEN RowCounts IS NULL THEN N'Lỗi đọc dữ liệu'
        WHEN RowCounts = 0 THEN N'Bảng không có dữ liệu'
        WHEN NonNullCount = 0 THEN N'Cột toàn NULL'
        WHEN NonNullCount < RowCounts THEN N'Cột có NULL'
        WHEN NonNullCount = RowCounts THEN N'Cột đầy đủ dữ liệu'
    END AS [DataStatus], -- Đã thêm dấu phẩy sửa lỗi cú pháp ở đây
    ISNULL(MinValue, N'N/A') AS [MinValue],
    ISNULL(MaxValue, N'N/A') AS [MaxValue],
    ISNULL(SampleValues, N'N/A') AS [SampleValues]
FROM #FinalReport
ORDER BY
    DatabaseName,
    SchemaName,
    TableName,
    ColumnName;

-- Dọn dẹp bảng tạm
DROP TABLE #FinalReport;