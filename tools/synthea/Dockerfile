# Build from the repository root through compose.yaml.
FROM eclipse-temurin:17-jre-jammy AS download

# Pin the upstream release and verify the published asset's SHA-256.
ARG SYNTHEA_VERSION=4.0.0
ARG SYNTHEA_SHA256=ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /download \
    && curl --fail --location --retry 3 \
       "https://github.com/synthetichealth/synthea/releases/download/v${SYNTHEA_VERSION}/synthea-with-dependencies.jar" \
       --output /download/synthea.jar \
    && echo "${SYNTHEA_SHA256}  /download/synthea.jar" | sha256sum --check --strict

FROM eclipse-temurin:17-jre-jammy
WORKDIR /opt/synthea
COPY --from=download --chmod=0444 /download/synthea.jar ./synthea.jar
RUN mkdir -p /output && chown 10001:10001 /output
USER 10001:10001
ENV TZ=UTC
ENTRYPOINT ["java", "-Duser.timezone=UTC", "-jar", "/opt/synthea/synthea.jar"]
CMD ["-p", "100", "-s", "42", "-cs", "42", "-r", "20250101", "-e", "20250101", "--exporter.baseDirectory=/output", "--exporter.fhir.export=true", "--exporter.csv.export=false"]
