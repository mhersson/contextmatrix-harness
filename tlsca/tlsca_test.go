package tlsca_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mhersson/contextmatrix-harness/tlsca"
)

// caFixture is a self-signed CA plus a leaf certificate it signs for 127.0.0.1,
// used to prove a client built from the CA PEM trusts a server presenting the
// leaf while a default client does not.
type caFixture struct {
	caPEMPath  string
	serverCert tls.Certificate
}

// newCAFixture generates the CA + leaf and writes the CA PEM to a temp file.
func newCAFixture(t *testing.T) caFixture {
	t.Helper()

	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)

	caTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "contextmatrix-test-ca"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}

	caDER, err := x509.CreateCertificate(rand.Reader, caTmpl, caTmpl, &caKey.PublicKey, caKey)
	require.NoError(t, err)

	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)

	leafTmpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: "127.0.0.1"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
	}

	caCert, err := x509.ParseCertificate(caDER)
	require.NoError(t, err)

	leafDER, err := x509.CreateCertificate(rand.Reader, leafTmpl, caCert, &leafKey.PublicKey, caKey)
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "ca.pem")
	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER})
	require.NoError(t, os.WriteFile(path, caPEM, 0o600))

	return caFixture{
		caPEMPath: path,
		serverCert: tls.Certificate{
			Certificate: [][]byte{leafDER},
			PrivateKey:  leafKey,
			Leaf:        leafCertOrNil(leafDER),
		},
	}
}

func leafCertOrNil(der []byte) *x509.Certificate {
	c, err := x509.ParseCertificate(der)
	if err != nil {
		return nil
	}

	return c
}

func TestCATransport(t *testing.T) {
	t.Run("empty path returns a nil transport", func(t *testing.T) {
		tr, err := tlsca.CATransport("")
		require.NoError(t, err)
		assert.Nil(t, tr, "empty path lets callers keep their default RoundTripper")
	})

	t.Run("valid PEM yields a client that trusts the CA-signed server", func(t *testing.T) {
		fx := newCAFixture(t)

		srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		}))
		srv.TLS = &tls.Config{Certificates: []tls.Certificate{fx.serverCert}, MinVersion: tls.VersionTLS12}

		srv.StartTLS()
		defer srv.Close()

		// A default client must reject the server (unknown authority).
		_, err := (&http.Client{}).Get(srv.URL)
		require.Error(t, err, "default client must not trust the private CA")

		// A client on the CA transport must accept it.
		tr, err := tlsca.CATransport(fx.caPEMPath)
		require.NoError(t, err)

		resp, err := (&http.Client{Transport: tr}).Get(srv.URL)
		require.NoError(t, err, "CA client must trust the CA-signed server")

		_ = resp.Body.Close()
		assert.Equal(t, http.StatusNoContent, resp.StatusCode)
	})

	t.Run("missing file returns an error", func(t *testing.T) {
		_, err := tlsca.CATransport(filepath.Join(t.TempDir(), "nope.pem"))
		require.Error(t, err)
	})

	t.Run("valid PEM yields a transport with the CA pool and TLS 1.2 floor", func(t *testing.T) {
		fx := newCAFixture(t)

		tr, err := tlsca.CATransport(fx.caPEMPath)
		require.NoError(t, err)
		require.NotNil(t, tr)
		require.NotNil(t, tr.TLSClientConfig)
		require.NotNil(t, tr.TLSClientConfig.RootCAs)
		assert.Equal(t, uint16(tls.VersionTLS12), tr.TLSClientConfig.MinVersion)
		// Proxy support is preserved (cloned from the default transport).
		assert.NotNil(t, tr.Proxy)
	})

	t.Run("bad PEM returns an error", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.pem")
		require.NoError(t, os.WriteFile(path, []byte("nope"), 0o600))
		_, err := tlsca.CATransport(path)
		require.Error(t, err)
	})
}

func TestHTTPClientWithCA(t *testing.T) {
	// A TLS server with a self-signed cert the system trust store rejects. Its
	// own certificate, written to a PEM, is the "extra CA" under test.
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	certPath := filepath.Join(t.TempDir(), "ca.pem")
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: srv.Certificate().Raw})
	require.NoError(t, os.WriteFile(certPath, pemBytes, 0o600))

	t.Run("empty path returns a plain client", func(t *testing.T) {
		c, err := tlsca.HTTPClientWithCA("")
		require.NoError(t, err)
		require.NotNil(t, c)
		assert.Nil(t, c.Transport, "empty path must not install a custom transport")
	})

	t.Run("valid CA is trusted", func(t *testing.T) {
		// Control: the default trust store must reject the self-signed server.
		_, err := http.Get(srv.URL) //nolint:bodyclose // request fails before a body exists
		require.Error(t, err, "control: default client must not trust the self-signed server")

		c, err := tlsca.HTTPClientWithCA(certPath)
		require.NoError(t, err)
		require.NotNil(t, c.Transport, "a CA path must install a custom transport")

		resp, err := c.Get(srv.URL)
		require.NoError(t, err, "the CA client must trust the server signed by the extra CA")

		defer func() { _ = resp.Body.Close() }()

		assert.Equal(t, http.StatusOK, resp.StatusCode)
	})

	t.Run("bad PEM errors", func(t *testing.T) {
		badPath := filepath.Join(t.TempDir(), "bad.pem")
		require.NoError(t, os.WriteFile(badPath, []byte("not a certificate"), 0o600))

		_, err := tlsca.HTTPClientWithCA(badPath)
		require.Error(t, err)
	})

	t.Run("missing file errors", func(t *testing.T) {
		_, err := tlsca.HTTPClientWithCA(filepath.Join(t.TempDir(), "does-not-exist.pem"))
		require.Error(t, err)
	})
}
